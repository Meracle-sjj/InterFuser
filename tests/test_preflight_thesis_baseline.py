"""
[INPUT]: 依赖 tools.evaluation.preflight_thesis_baseline 的 P0 API，并在临时仓库中构造 checkpoint、路线、场景和哈希配置。
[OUTPUT]: 提供 M0 输入哈希、路线集合分区、场景覆盖和地图排除策略的回归测试。
[POS]: tests 的闭环评测启动门禁测试，保证昂贵 CARLA 运行只消费完整且未漂移的版本化输入。
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from tools.evaluation.preflight_thesis_baseline import preflight_baseline


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class ThesisBaselinePreflightTests(unittest.TestCase):
    def _fixture(self, root, scenario_events=1):
        root = Path(root)
        inputs = root / "inputs"
        inputs.mkdir()
        routes = inputs / "routes.xml"
        routes.write_text(
            "<routes>"
            "<route id='0' town='Town01'><waypoint/><waypoint/></route>"
            "<route id='1' town='Town06'><waypoint/><waypoint/></route>"
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
                                {
                                    "scenario_type": "Scenario1",
                                    "available_event_configurations": [
                                        {} for _ in range(scenario_events)
                                    ],
                                }
                            ],
                            "Town06": [
                                {
                                    "scenario_type": "Scenario1",
                                    "available_event_configurations": [{}],
                                }
                            ],
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        agent = inputs / "agent.py"
        agent.write_text("AGENT = True\n", encoding="utf-8")
        agent_config = inputs / "agent_config.py"
        agent_config.write_text("CONFIG = True\n", encoding="utf-8")
        checkpoint = root / "model.tar"
        checkpoint.write_bytes(b"checkpoint")
        config = {
            "schema_version": 1,
            "code_anchor": "unused-in-unit-test",
            "checkpoint": {
                "path": str(checkpoint),
                "sha256": _sha(checkpoint),
            },
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
            },
            "background_vehicles_by_route": {"0": 10},
            "random_seeds": [0, 1, 2],
            "result_root": "results/thesis_m0",
        }
        config_path = root / "baseline.json"
        config_path.write_text(json.dumps(config), encoding="utf-8")
        return config_path

    def test_accepts_complete_static_contract(self):
        with tempfile.TemporaryDirectory() as root:
            config = self._fixture(root)

            report = preflight_baseline(config, root, check_git=False)

        self.assertTrue(report["valid"])
        self.assertEqual(report["route_count"], 2)
        self.assertEqual(report["development_routes"], [0])
        self.assertEqual(report["excluded_routes"], [1])
        self.assertEqual(len(report["warnings"]), 1)

    def test_rejects_input_hash_drift(self):
        with tempfile.TemporaryDirectory() as root:
            config = self._fixture(root)
            data = json.loads(config.read_text())
            data["inputs"]["agent"]["sha256"] = "0" * 64
            config.write_text(json.dumps(data), encoding="utf-8")

            report = preflight_baseline(config, root, check_git=False)

        self.assertFalse(report["valid"])
        self.assertTrue(any("agent sha256 mismatch" in item for item in report["errors"]))

    def test_rejects_route_partition_gap(self):
        with tempfile.TemporaryDirectory() as root:
            config = self._fixture(root)
            data = json.loads(config.read_text())
            data["route_sets"]["excluded_until_map_install"] = []
            config.write_text(json.dumps(data), encoding="utf-8")

            report = preflight_baseline(config, root, check_git=False)

        self.assertFalse(report["valid"])
        self.assertTrue(
            any("route partition mismatch" in item for item in report["errors"])
        )

    def test_rejects_route_town_without_scenario_events(self):
        with tempfile.TemporaryDirectory() as root:
            config = self._fixture(root, scenario_events=0)

            report = preflight_baseline(config, root, check_git=False)

        self.assertFalse(report["valid"])
        self.assertIn(
            "scenario file has no events for route town Town01", report["errors"]
        )


if __name__ == "__main__":
    unittest.main()
