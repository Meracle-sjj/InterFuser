"""
[INPUT]: 依赖 tools.evaluation.run_thesis_baseline 的计划、路线拆分、端口门禁和 Leaderboard 结果解析 API，并使用临时配置构造最小 P0 合法输入。
[OUTPUT]: 提供 D7 route/seed 选择、GPU 覆盖留痕、单路线 XML、驾驶失败保留和端口占用拒绝的回归测试。
[POS]: tests 的 M0 runner 纯逻辑测试，不启动 CARLA；外部进程生命周期由真实单路线 smoke 进一步验证。
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

import hashlib
import json
import socket
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

from tools.evaluation.run_thesis_baseline import (
    RunnerError,
    build_run_plan,
    ensure_ports_free,
    parse_leaderboard_result,
    write_single_route_xml,
)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class ThesisBaselineRunnerTests(unittest.TestCase):
    def _fixture(self, root):
        root = Path(root)
        inputs = root / "inputs"
        inputs.mkdir()
        routes = inputs / "routes.xml"
        routes.write_text(
            "<routes>"
            "<route id='0' town='Town01'><waypoint x='1'/><waypoint x='2'/></route>"
            "<route id='1' town='Town06'><waypoint x='3'/><waypoint x='4'/></route>"
            "</routes>",
            encoding="utf-8",
        )
        scenarios = inputs / "scenarios.json"
        scenarios.write_text(
            json.dumps(
                {
                    "available_scenarios": [
                        {
                            "Town01": [
                                {"available_event_configurations": [{}]}
                            ],
                            "Town06": [
                                {"available_event_configurations": [{}]}
                            ],
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        agent = inputs / "agent.py"
        agent.write_text("AGENT=True\n", encoding="utf-8")
        agent_config = inputs / "agent_config.py"
        agent_config.write_text("CONFIG=True\n", encoding="utf-8")
        checkpoint = root / "model.tar"
        checkpoint.write_bytes(b"checkpoint")
        config = {
            "schema_version": 1,
            "code_anchor": "unused",
            "runtime_code_roots": ["inputs"],
            "runtime": {
                "agent_cuda_visible_device": 6,
                "carla_graphics_adapter": 7,
                "carla_port": 2155,
                "traffic_manager_port": 2255,
                "carla_start_timeout_seconds": 30,
                "carla_client_timeout_seconds": 10,
                "external_route_timeout_seconds": 100,
                "carla_provider_seed_offset": 2000,
                "gpu_busy_memory_threshold_mb": 1024,
            },
            "checkpoint": {"path": str(checkpoint), "sha256": _sha(checkpoint)},
            "inputs": {
                "routes": {"path": "inputs/routes.xml", "sha256": _sha(routes)},
                "scenarios": {
                    "path": "inputs/scenarios.json",
                    "sha256": _sha(scenarios),
                },
                "agent": {"path": "inputs/agent.py", "sha256": _sha(agent)},
                "agent_config": {
                    "path": "inputs/agent_config.py",
                    "sha256": _sha(agent_config),
                },
            },
            "route_sets": {
                "development_d7": [0],
                "primary_a36": [0],
                "excluded_until_map_install": [1],
            },
            "map_policy": {
                "available_route_towns": ["Town01"],
                "unavailable_route_towns": ["Town06"],
                "allow_opt_runtime_equivalent": True,
            },
            "result_root": "results/thesis_m0",
            "background_vehicles_by_town": {"Town01": 10},
            "random_seeds": [0, 1, 2],
            "environment": {"INTERFUSER_REUSE_CURRENT_WORLD": "1"},
        }
        config_path = root / "baseline.json"
        config_path.write_text(json.dumps(config), encoding="utf-8")
        return config_path, routes

    def test_build_plan_records_route_seed_and_hardware_overrides(self):
        with tempfile.TemporaryDirectory() as root:
            config, _ = self._fixture(root)

            plan = build_run_plan(
                config,
                root,
                run_id="b0-d7-smoke",
                route_ids=[0],
                seeds=[1],
                agent_gpu=2,
                carla_graphics_adapter=3,
                timeout_seconds=60,
                check_git=False,
            )

        self.assertEqual(plan["runtime"]["agent_cuda_visible_device"], 2)
        self.assertEqual(plan["runtime"]["carla_graphics_adapter"], 3)
        self.assertEqual(len(plan["runner_sha256"]), 64)
        self.assertTrue(plan["python_executable"])
        self.assertEqual(len(plan["attempts"]), 1)
        attempt = plan["attempts"][0]
        self.assertEqual(attempt["attempt_id"], "route_00_seed_1")
        self.assertEqual(attempt["runtime_map"], "Town01_Opt")
        self.assertEqual(attempt["background_vehicles"], 10)
        self.assertEqual(attempt["carla_provider_seed"], 2001)
        self.assertEqual(attempt["external_timeout_seconds"], 60)

    def test_rejects_run_id_path_traversal(self):
        with tempfile.TemporaryDirectory() as root:
            config, _ = self._fixture(root)
            with self.assertRaisesRegex(RunnerError, "run_id"):
                build_run_plan(config, root, run_id="../escape", check_git=False)

    def test_single_route_xml_preserves_original_town_and_waypoints(self):
        with tempfile.TemporaryDirectory() as root:
            _, routes = self._fixture(root)
            output = Path(root) / "single.xml"

            write_single_route_xml(routes, 0, output)

            parsed = ET.parse(output).getroot().findall("route")
        self.assertEqual(len(parsed), 1)
        self.assertEqual(parsed[0].attrib, {"id": "0", "town": "Town01"})
        self.assertEqual([item.attrib["x"] for item in parsed[0].findall("waypoint")], ["1", "2"])

    def test_parser_accepts_driving_failure_with_complete_scores(self):
        record = {
            "_checkpoint": {
                "records": [
                    {
                        "route_id": "RouteScenario_0",
                        "status": "Failed - Agent got blocked",
                        "scores": {
                            "score_composed": 10.0,
                            "score_route": 20.0,
                            "score_penalty": 0.5,
                        },
                        "infractions": {"vehicle_blocked": ["blocked"]},
                        "meta": {"route_length": 100.0, "duration_game": 10.0},
                    }
                ]
            },
            "entry_status": "Finished with agent errors",
            "eligible": True,
        }
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "result.json"
            path.write_text(json.dumps(record), encoding="utf-8")

            parsed = parse_leaderboard_result(path)

        self.assertTrue(parsed["valid"])
        self.assertEqual(parsed["status"], "Failed - Agent got blocked")
        self.assertEqual(parsed["scores"]["score_composed"], 10.0)
        self.assertEqual(parsed["infraction_counts"]["vehicle_blocked"], 1)

    def test_parser_rejects_agent_setup_failure_even_with_scores(self):
        record = {
            "_checkpoint": {
                "records": [
                    {
                        "route_id": "RouteScenario_0",
                        "status": "Failed - Agent couldn't be set up",
                        "scores": {
                            "score_composed": 0.0,
                            "score_route": 0.0,
                            "score_penalty": 1.0,
                        },
                        "infractions": {},
                        "meta": {"route_length": 100.0, "duration_game": 0.0},
                    }
                ]
            }
        }
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "result.json"
            path.write_text(json.dumps(record), encoding="utf-8")

            parsed = parse_leaderboard_result(path)

        self.assertFalse(parsed["valid"])
        self.assertIn("couldn't be set up", parsed["error"])
        self.assertIn("no positive game duration", parsed["error"])

    def test_port_gate_rejects_listener(self):
        listener = socket.socket()
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        port = listener.getsockname()[1]
        try:
            with self.assertRaisesRegex(RunnerError, str(port)):
                ensure_ports_free([port])
        finally:
            listener.close()


if __name__ == "__main__":
    unittest.main()
