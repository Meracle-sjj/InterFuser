import os
import json
import datetime
import pathlib
import time
import imp
import cv2
import carla
from collections import deque

import torch
import torch.nn as nn
import carla
import numpy as np
from PIL import Image
from easydict import EasyDict
from sklearn.decomposition import PCA

from torchvision import transforms
from leaderboard.autoagents import autonomous_agent
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from timm.models import create_model
from team_code.utils import lidar_to_histogram_features, transform_2d_points
from team_code.planner import RoutePlanner
from team_code.interfuser_controller import InterfuserController
from team_code.render import render, render_self_car, render_waypoints
from team_code.tracker import Tracker

import math
import yaml

try:
    import pygame
except ImportError:
    raise RuntimeError("cannot import pygame, make sure pygame package is installed")


class LinearClassifier(nn.Module):
    def __init__(self, in_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, 1)

    def forward(self, x):
        return self.linear(x)


SAVE_PATH = os.environ.get("SAVE_PATH", 'eval')
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


class DisplayInterface(object):
    def __init__(self):
        self._width = 1200
        self._height = 800
        self._surface = None
        
        # 检查是否禁用窗口显示（但保持渲染逻辑）
        self._headless = os.environ.get('INTERFUSER_HEADLESS', '0') == '1'
        
        if not self._headless:
            pygame.init()
            pygame.font.init()
            self._clock = pygame.time.Clock()
            self._display = pygame.display.set_mode(
                (self._width, self._height), pygame.HWSURFACE | pygame.DOUBLEBUF
            )
            pygame.display.set_caption("Human Agent")

    def run_interface(self, input_data):
        rgb = input_data['rgb']
        rgb_left = input_data['rgb_left']
        rgb_right = input_data['rgb_right']
        rgb_focus = input_data['rgb_focus']
        map = input_data['map']
        surface = np.zeros((800, 1200, 3),np.uint8)
        surface[0:600, 0:800] = rgb
        surface[0:400,800:1200] = map
        surface[400:600,800:1000] = input_data['map_t1']
        surface[400:600,1000:1200] = input_data['map_t2']
        surface[0:150,:200] = input_data['rgb_left']
        surface[0:150, 600:800] = input_data['rgb_right']
        surface[0:150, 325:475] = input_data['rgb_focus']
        surface = cv2.putText(surface, input_data['control'], (20,580), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255), 1)
        surface = cv2.putText(surface, input_data['meta_infos'][0], (20,560), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255), 1)
        surface = cv2.putText(surface, input_data['meta_infos'][1], (20,540), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255), 1)
        surface = cv2.putText(surface, input_data['time'], (20,520), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255), 1)
        surface = cv2.putText(surface, input_data.get('junction_prob', 'N/A'), (20,500), cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0), 1)
        surface = cv2.putText(surface, 'Left  View', (40,135), cv2.FONT_HERSHEY_SIMPLEX,0.75,(0,0,0), 2)
        surface = cv2.putText(surface, 'Focus View', (335,135), cv2.FONT_HERSHEY_SIMPLEX,0.75,(0,0,0), 2)
        surface = cv2.putText(surface, 'Right View', (640,135), cv2.FONT_HERSHEY_SIMPLEX,0.75,(0,0,0), 2)
        surface = cv2.putText(surface,f"RoutePlanner Command: {input_data.get('next_command_name', 'N/A')}",(20, 595),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0, 255, 0),2)            
        surface = cv2.putText(surface, 'Future Prediction', (940,420), cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0), 2)
        surface = cv2.putText(surface, 't', (1160,385), cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,0,0), 2)
        surface = cv2.putText(surface, '0', (1170,385), cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0), 2)
        surface = cv2.putText(surface, 't', (960,585), cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,0,0), 2)
        surface = cv2.putText(surface, '1', (970,585), cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0), 2)
        surface = cv2.putText(surface, 't', (1160,585), cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,0,0), 2)
        surface = cv2.putText(surface, '2', (1170,585), cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0), 2)
        if 'meta_map' in input_data:
            # cv2.imshow('Meta Map', input_data['meta_map'])
            # cv2.waitKey(1)
            surface[600:800, 0:200] = input_data['meta_map']  
            surface = cv2.putText(surface, 'traffic_meta[0]', (10, 790), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        
        if 'token_patch' in input_data:
            surface[600:800, 200:400] = input_data['token_patch']
            surface = cv2.putText(surface, 'token_rgb', (210, 790), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            
        surface[:150,198:202]=0
        surface[:150,323:327]=0
        surface[:150,473:477]=0
        surface[:150,598:602]=0
        surface[148:152, :200] = 0
        surface[148:152, 325:475] = 0
        surface[148:152, 600:800] = 0
        surface[430:600, 998:1000] = 255
        surface[0:600, 798:800] = 255
        surface[0:600, 1198:1200] = 255
        surface[0:2, 800:1200] = 255
        surface[598:600, 800:1200] = 255
        surface[398:400, 800:1200] = 255


        # display image - 只在非 headless 模式下更新窗口
        if not self._headless:
            self._surface = pygame.surfarray.make_surface(surface.swapaxes(0, 1))
            if self._surface is not None:
                self._display.blit(self._surface, (0, 0))
            pygame.display.flip()
            pygame.event.get()
        
        return surface

    def _quit(self):
        if not self._headless:
            pygame.quit()


def get_entry_point():
    return "InterfuserAgent"


class Resize2FixedSize:
    def __init__(self, size):
        self.size = size

    def __call__(self, pil_img):
        pil_img = pil_img.resize(self.size)
        return pil_img


def create_carla_rgb_transform(
    input_size, need_scale=True, mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD
):

    if isinstance(input_size, (tuple, list)):
        img_size = input_size[-2:]
    else:
        img_size = input_size
    tfl = []

    if isinstance(input_size, (tuple, list)):
        input_size_num = input_size[-1]
    else:
        input_size_num = input_size

    if need_scale:
        if input_size_num == 112:
            tfl.append(Resize2FixedSize((170, 128)))
        elif input_size_num == 128:
            tfl.append(Resize2FixedSize((195, 146)))
        elif input_size_num == 224:
            tfl.append(Resize2FixedSize((341, 256)))
        elif input_size_num == 256:
            tfl.append(Resize2FixedSize((288, 288)))
        else:
            raise ValueError("Can't find proper crop size")
    tfl.append(transforms.CenterCrop(img_size))
    tfl.append(transforms.ToTensor())
    tfl.append(transforms.Normalize(mean=torch.tensor(mean), std=torch.tensor(std)))

    return transforms.Compose(tfl)


class InterfuserAgent(autonomous_agent.AutonomousAgent):
    def setup(self, path_to_conf_file):

        self._hic = DisplayInterface()
        self.lidar_processed = list()
        self.track = autonomous_agent.Track.SENSORS
        self.step = -1
        self.wall_start = time.time()
        self.initialized = False
        self.rgb_front_transform = create_carla_rgb_transform(224)
        self.rgb_left_transform = create_carla_rgb_transform(128)
        self.rgb_right_transform = create_carla_rgb_transform(128)
        self.rgb_center_transform = create_carla_rgb_transform(128, need_scale=False)
        self.pca_token_buffer = []

        self.tracker = Tracker()

        self.input_buffer = {
            "rgb": deque(),
            "rgb_left": deque(),
            "rgb_right": deque(),
            "rgb_rear": deque(),
            "lidar": deque(),
            "gps": deque(),
            "thetas": deque(),
        }

        self.config = imp.load_source("MainModel", path_to_conf_file).GlobalConfig()
        self.skip_frames = self.config.skip_frames
        self.controller = InterfuserController(self.config)
        if isinstance(self.config.model, list):
            self.ensemble = True
        else:
            self.ensemble = False

        if self.ensemble:
            for i in range(len(self.config.model)):
                self.nets = []
                net = create_model(self.config.model[i])
                path_to_model_file = self.config.model_path[i]
                print('load model: %s' % path_to_model_file)
                net.load_state_dict(torch.load(path_to_model_file, weights_only=False)["state_dict"])
                net.cuda()
                net.eval()
                self.nets.append(net)
        else:
            self.net = create_model(self.config.model)
            path_to_model_file = self.config.model_path
            print('load model: %s' % path_to_model_file)
            self.net.load_state_dict(torch.load(path_to_model_file, weights_only=False)["state_dict"])
            self.net.cuda()
            self.net.eval()

        # Load external junction model
            # Load external junction model
            # Priority: env var JUNCTION_MODEL_PATH -> common local paths -> error
            junction_model_path = os.environ.get('JUNCTION_MODEL_PATH')
            if not junction_model_path:
                # Try a few sensible defaults within this repo
                repo_root = pathlib.Path(__file__).resolve().parents[2]
                candidates = [
                    repo_root / 'model_N=500_old.pt',
                    repo_root / 'cb_loss_result' / 'model_N=500.pt',
                    pathlib.Path('/home/shijj/interfuser/model_N=500_old.pt'),
                ]
                for c in candidates:
                    if c.exists():
                        junction_model_path = str(c)
                        break
            if not junction_model_path or not pathlib.Path(junction_model_path).exists():
                raise FileNotFoundError(
                    'Junction model checkpoint not found. Set env JUNCTION_MODEL_PATH to the .pt file, '
                    f'for example: {repo_root}/model_N=500_old.pt'
                )
            checkpoint = torch.load(junction_model_path, map_location='cpu', weights_only=False)
        self.junction_model = LinearClassifier(checkpoint['in_features'])
        self.junction_model.load_state_dict(checkpoint['state_dict'])
        self.junction_model.eval()
        self.scaler_mean = torch.tensor(checkpoint['scaler_mean'], dtype=torch.float32).cuda()
        self.scaler_scale = torch.tensor(checkpoint['scaler_scale'], dtype=torch.float32).cuda()
        self.junction_model.cuda()
        self.softmax = torch.nn.Softmax(dim=1)
        self.traffic_meta_moving_avg = np.zeros((400, 7))
        self.momentum = self.config.momentum
        self.prev_lidar = None
        self.prev_control = None
        self.prev_surround_map = None

        self.save_path = None
        if SAVE_PATH is not None:
            now = datetime.datetime.now()
            string = pathlib.Path(os.environ["ROUTES"]).stem + "_"
            string += "_".join(
                map(
                    lambda x: "%02d" % x,
                    (now.month, now.day, now.hour, now.minute, now.second),
                )
            )

            print(string)

            self.save_path = pathlib.Path(SAVE_PATH) / string
            self.save_path.mkdir(parents=True, exist_ok=False)
            (self.save_path / "meta").mkdir(parents=True, exist_ok=False)

    def _init(self):
        self._route_planner = RoutePlanner(7.5, 25.0)
        self._route_planner.set_route(self._global_plan, True)
        self.initialized = True

    def _get_position(self, tick_data):
        gps_raw = tick_data["gps"]  # [latitude, longitude] in degrees from CARLA GNSS sensor
        
        # CARLA GNSS outputs GPS that needs same conversion as planner routes
        # Both use: [lon, lat] → [x, y] with scale but NO negation
        # (The negation in planner.set_route is for leaderboard GPS, not CARLA GNSS)
        gps = np.array([gps_raw[1], gps_raw[0]])
        gps = (gps - self._route_planner.mean) * self._route_planner.scale
        
        return gps

    def _get_lane_debug(self):
        try:
            ego = CarlaDataProvider.get_hero_actor()
            if ego is None:
                return {}
            vehicle_transform = ego.get_transform()
            vehicle_location = vehicle_transform.location
            waypoint = CarlaDataProvider.get_map().get_waypoint(
                vehicle_location, lane_type=carla.LaneType.Driving, project_to_road=True
            )
            if waypoint is None:
                return {}
            lane_location = waypoint.transform.location
            right_vector = waypoint.transform.get_right_vector()
            dx = vehicle_location.x - lane_location.x
            dy = vehicle_location.y - lane_location.y
            lane_offset = dx * right_vector.x + dy * right_vector.y
            return {
                "lane_offset": float(lane_offset),
                "lane_width": float(waypoint.lane_width),
                "road_id": int(waypoint.road_id),
                "lane_id": int(waypoint.lane_id),
            }
        except Exception as exc:
            if self.step % 100 == 0:
                print(f"[WARN] lane debug unavailable at step {self.step}: {exc}", flush=True)
            return {}

    def sensors(self):
        return [
            {
                "type": "sensor.camera.rgb",
                "x": 1.3,
                "y": 0.0,
                "z": 2.3,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 0.0,
                "width": 800,
                "height": 600,
                "fov": 100,
                "id": "rgb",
            },
            {
                "type": "sensor.camera.rgb",
                "x": 1.3,
                "y": 0.0,
                "z": 2.3,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": -60.0,
                "width": 400,
                "height": 300,
                "fov": 100,
                "id": "rgb_left",
            },
            {
                "type": "sensor.camera.rgb",
                "x": 1.3,
                "y": 0.0,
                "z": 2.3,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 60.0,
                "width": 400,
                "height": 300,
                "fov": 100,
                "id": "rgb_right",
            },
            {
                "type": "sensor.lidar.ray_cast",
                "x": 1.3,
                "y": 0.0,
                "z": 2.5,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": -90.0,
                "id": "lidar",
            },
            {
                "type": "sensor.other.imu",
                "x": 0.0,
                "y": 0.0,
                "z": 0.0,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 0.0,
                "sensor_tick": 0.05,
                "id": "imu",
            },
            {
                "type": "sensor.other.gnss",
                "x": 0.0,
                "y": 0.0,
                "z": 0.0,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 0.0,
                "sensor_tick": 0.01,
                "id": "gps",
            },
            {"type": "sensor.speedometer", "reading_frequency": 20, "id": "speed"},
        ]

    def tick(self, input_data):

        rgb = cv2.cvtColor(input_data["rgb"][1][:, :, :3], cv2.COLOR_BGR2RGB)
        rgb_left = cv2.cvtColor(input_data["rgb_left"][1][:, :, :3], cv2.COLOR_BGR2RGB)
        rgb_right = cv2.cvtColor(
            input_data["rgb_right"][1][:, :, :3], cv2.COLOR_BGR2RGB
        )
        gps = input_data["gps"][1][:2]
        speed = input_data["speed"][1]["speed"]
        compass = input_data["imu"][1][-1]
        if (
            math.isnan(compass) == True
        ):  # It can happen that the compass sends nan for a few frames
            compass = 0.0

        result = {
            "rgb": rgb,
            "rgb_left": rgb_left,
            "rgb_right": rgb_right,
            "gps": gps,
            "speed": speed,
            "compass": compass,
        }

        pos = self._get_position(result)

        lidar_data = input_data['lidar'][1]
        result['raw_lidar'] = lidar_data

        lidar_unprocessed = lidar_data[:, :3]
        lidar_unprocessed[:, 1] *= -1
        full_lidar = transform_2d_points(
            lidar_unprocessed,
            np.pi / 2 - compass,
            -pos[0],
            -pos[1],
            np.pi / 2 - compass,
            -pos[0],
            -pos[1],
        )
        lidar_processed = lidar_to_histogram_features(full_lidar, crop=224)
        if self.step % 2 == 0 or self.step < 4:
            self.prev_lidar = lidar_processed
        result["lidar"] = self.prev_lidar

        result["gps"] = pos
        next_wp, next_cmd= self._route_planner.run_step(pos)
        next_cmd_name = next_cmd.name
        result["next_command"] = next_cmd.value
        result["next_command_name"] = next_cmd_name
        result['measurements'] = [pos[0], pos[1], compass, speed]

        # 计算世界坐标系下的相对位置
        local_command_point = np.array([next_wp[0] - pos[0], next_wp[1] - pos[1]])
        
        # 转换到车辆局部坐标系
        theta = compass + np.pi / 2
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        local_command_point = R.T.dot(local_command_point)
        
        # CARLA 0.9.16 距离缩放修复（如果需要）
        target_distance = np.linalg.norm(local_command_point)
        if target_distance > 100:
            local_command_point = local_command_point * 0.1

        result["target_point_raw"] = local_command_point.copy()
        target_lateral_gain = float(os.environ.get("INTERFUSER_TARGET_LATERAL_GAIN", "-1.0"))
        local_command_point[1] *= target_lateral_gain
        
        result["target_point"] = local_command_point

        return result

    @torch.no_grad()
    def run_step(self, input_data, timestamp):
        if not self.initialized:
            self._init()

        self.step += 1
        if self.step % self.skip_frames != 0 and self.step > 4:
            return self.prev_control

        tick_data = self.tick(input_data)

        velocity = tick_data["speed"]
        command = tick_data["next_command"]

        rgb = (
            self.rgb_front_transform(Image.fromarray(tick_data["rgb"]))
            .unsqueeze(0)
            .cuda()
            .float()
        )
        rgb_left = (
            self.rgb_left_transform(Image.fromarray(tick_data["rgb_left"]))
            .unsqueeze(0)
            .cuda()
            .float()
        )
        rgb_right = (
            self.rgb_right_transform(Image.fromarray(tick_data["rgb_right"]))
            .unsqueeze(0)
            .cuda()
            .float()
        )
        rgb_center = (
            self.rgb_center_transform(Image.fromarray(tick_data["rgb"]))
            .unsqueeze(0)
            .cuda()
            .float()
        )

        cmd_one_hot = [0, 0, 0, 0, 0, 0]
        cmd = command - 1
        cmd_one_hot[cmd] = 1
        cmd_one_hot.append(velocity)
        mes = np.array(cmd_one_hot)
        mes = torch.from_numpy(mes).float().unsqueeze(0).cuda()

        input_data = {}
        input_data["rgb"] = rgb
        input_data["rgb_left"] = rgb_left
        input_data["rgb_right"] = rgb_right
        input_data["rgb_center"] = rgb_center
        input_data["measurements"] = mes
        input_data["target_point"] = (
            torch.from_numpy(tick_data["target_point"]).float().cuda().view(1, -1)
        )
        input_data["lidar"] = (
            torch.from_numpy(tick_data["lidar"]).float().cuda().unsqueeze(0)
        )
        if self.ensemble:
            outputs = []
            with torch.no_grad():
                for net in self.nets:
                    output = net(input_data)
                    outputs.append(output)
            traffic_meta = torch.mean(torch.stack([x[0] for x in outputs]), 0)
            pred_waypoints = torch.mean(torch.stack([x[1] for x in outputs]), 0)
            is_junction = torch.mean(torch.stack([x[2] for x in outputs]), 0)
            traffic_light_state = torch.mean(torch.stack([x[3] for x in outputs]), 0)
            stop_sign = torch.mean(torch.stack([x[4] for x in outputs]), 0)
            bev_feature = torch.mean(torch.stack([x[5] for x in outputs]), 0)
        else:
            with torch.no_grad():
                (
                    traffic_meta,
                    pred_waypoints,
                    is_junction,
                    traffic_light_state,
                    stop_sign,
                    bev_feature,
                    traffic_state_feature
                ) = self.net(input_data)
        traffic_meta = traffic_meta.detach().cpu().numpy()[0]
        bev_feature = bev_feature.detach().cpu().numpy()[0]
        pred_waypoints = pred_waypoints.detach().cpu().numpy()[0]
        
        is_junction = self.softmax(is_junction).detach().cpu().numpy().reshape(-1)[0]
        traffic_light_state = (
            self.softmax(traffic_light_state).detach().cpu().numpy().reshape(-1)[0]
        )
        stop_sign = self.softmax(stop_sign).detach().cpu().numpy().reshape(-1)[0]

        # Handle traffic_state_feature: if it's 3D [1, 400, 256], pool it to [1, 256]
        if traffic_state_feature.dim() == 3:
            # Apply mean pooling over the spatial dimension (400)
            traffic_state_feature_pooled = traffic_state_feature.mean(dim=1)  # [1, 256]
        else:
            traffic_state_feature_pooled = traffic_state_feature  # Already [1, 256]
        
        # 原始 token 是 (1, 256) 的张量
        token = traffic_state_feature_pooled
        if torch.is_tensor(token):
            token = token.cpu().numpy()
        # 转化为(256,) 的一维数组便于后续处理
        token = np.squeeze(token)  # 变成 (256,)

        # Compute junction prediction
        traffic_state_feature_tensor = traffic_state_feature_pooled.clone().detach().float().cuda()
        traffic_light_state_feature_std = (traffic_state_feature_tensor - self.scaler_mean) / self.scaler_scale
        logits = self.junction_model(traffic_light_state_feature_std)
        new_junction_prediction = torch.sigmoid(logits).item()

        # 存储 token 用于 PCA
        self.pca_token_buffer.append(token)
        # 如果 pca_token_buffer 中的 token 数量超过 50，则进行 PCA
        if not hasattr(self, 'token_pca') and len(self.pca_token_buffer) >= 50:
            self.token_pca = PCA(n_components=3)
            self.token_pca.fit(np.stack(self.pca_token_buffer)) #把50个token进行PCA降维学习
        # 可视化
        if hasattr(self, 'token_pca'):
            token_rgb = self.token_pca.transform(token.reshape(1, -1))[0] 
        else:
            token_rgb = token[:3]

        #归一化到0-1，并最终生成色块
        token_rgb = (token_rgb - np.min(token_rgb)) / (np.ptp(token_rgb) + 1e-6)
        token_rgb = (token_rgb * 255).astype(np.uint8)
        color_patch = np.ones((200, 200, 3), dtype=np.uint8) * token_rgb.reshape(1, 1, 3)
        tick_data["token_patch"] = color_patch

        #保留token用于后续处理
        csv_path = self.save_path / "token_redlight.csv"
        with open(csv_path, "a") as f:
            token_str = ",".join([f"{x:.6f}" for x in token])
            f.write(f"{self.step},{token_str},{traffic_light_state}\n")
        # 保存红灯概率到 redlight.csv
        redlight_csv_path = self.save_path / "redlight.csv"
        with open(redlight_csv_path, "a") as f:
            f.write(f"{self.step},{traffic_light_state}\n")

        # 保存 junction 概率到 junction.csv
        junction_csv_path = self.save_path / "junction.csv"
        with open(junction_csv_path, "a") as f:
            f.write(f"{self.step},{new_junction_prediction}\n")

        if self.step % 2 == 0 or self.step < 4:
            traffic_meta = self.tracker.update_and_predict(traffic_meta.reshape(20, 20, -1), tick_data['gps'], tick_data['compass'], self.step // 2)
            traffic_meta = traffic_meta.reshape(400, -1)
            self.traffic_meta_moving_avg = (
                self.momentum * self.traffic_meta_moving_avg
                + (1 - self.momentum) * traffic_meta
            )
        traffic_meta = self.traffic_meta_moving_avg

        tick_data["raw"] = traffic_meta
        tick_data["bev_feature"] = bev_feature

        # DEBUG: Check for NaNs
        if np.isnan(pred_waypoints).any():
            print(f"[ERROR] pred_waypoints contains NaNs at step {self.step}!", flush=True)
            print(f"Input target_point: {input_data['target_point']}", flush=True)
            print(f"Input measurements: {input_data['measurements']}", flush=True)

        pred_waypoints = pred_waypoints.reshape(-1, 2)
        control_waypoints = np.stack([pred_waypoints[:, 1], pred_waypoints[:, 0]], axis=1)
        lateral_gain = float(os.environ.get("INTERFUSER_WP_LATERAL_GAIN", "0.35"))
        control_waypoints[:, 0] *= lateral_gain
        lateral_bias = float(os.environ.get("INTERFUSER_WP_LATERAL_BIAS", "0.0"))
        control_waypoints[:, 0] += lateral_bias

        steer, throttle, brake, meta_infos = self.controller.run_step(
            velocity,
            control_waypoints,
            is_junction,
            traffic_light_state,
            stop_sign,
            self.traffic_meta_moving_avg,
            aux_junction=new_junction_prediction,
        )
        control_debug = meta_infos[4] if len(meta_infos) > 4 else {}
        lane_debug = self._get_lane_debug()

        steer_before_lane_center = float(steer)
        lane_center_correction = 0.0
        lane_offset = lane_debug.get("lane_offset")
        lane_center_gain = float(os.environ.get("INTERFUSER_LANE_CENTER_STEER_GAIN", "0.0"))
        if lane_center_gain != 0.0 and lane_offset is not None and velocity > 1.0 and brake is False:
            lane_center_deadband = float(os.environ.get("INTERFUSER_LANE_CENTER_DEADBAND", "0.15"))
            lane_center_max = float(os.environ.get("INTERFUSER_LANE_CENTER_MAX", "0.25"))
            if abs(lane_offset) > lane_center_deadband:
                lane_error = lane_offset - np.sign(lane_offset) * lane_center_deadband
                lane_center_correction = float(
                    np.clip(-lane_center_gain * lane_error, -lane_center_max, lane_center_max)
                )
                steer = np.clip(steer + lane_center_correction, -1.0, 1.0)
        control_debug["steer_before_lane_center"] = steer_before_lane_center
        control_debug["lane_center_correction"] = lane_center_correction

        if brake < 0.05:
            brake = 0.0
        if brake > 0.1:
            throttle = 0.0

        control = carla.VehicleControl()
        control.steer = float(steer)
        control.throttle = float(throttle)
        control.brake = float(brake)

        surround_map, box_info = render(traffic_meta.reshape(20, 20, 7), pixels_per_meter=20)
        surround_map = surround_map[:400, 160:560]
        surround_map = np.stack([surround_map, surround_map, surround_map], 2)

        self_car_map = render_self_car(
            loc=np.array([0, 0]),
            ori=np.array([0, -1]),
            box=np.array([2.45, 1.0]),
            color=[1, 1, 0], pixels_per_meter=20
        )[:400, 160:560]

        safe_index = 10
        for i in range(10):
            if control_waypoints[i, 0] ** 2 + control_waypoints[i, 1] ** 2> (meta_infos[3]+0.5) ** 2:
                safe_index = i
                break
        wp1 = render_waypoints(control_waypoints[:safe_index], pixels_per_meter=20, color=(0, 255, 0))[:400, 160:560]
        wp2 = render_waypoints(control_waypoints[safe_index:], pixels_per_meter=20, color=(255, 0, 0))[:400, 160:560]
        wp = wp1 + wp2

        surround_map = np.clip(
            (
                surround_map.astype(np.float32)
                + self_car_map.astype(np.float32)
                + wp.astype(np.float32)
            ),
            0,
            255,
        ).astype(np.uint8)

        map_t1, box_info = render(traffic_meta.reshape(20, 20, 7), pixels_per_meter=20, t=1)
        map_t1 = map_t1[:400, 160:560]
        map_t1 = np.stack([map_t1, map_t1, map_t1], 2)
        map_t1 = np.clip(map_t1.astype(np.float32) + self_car_map.astype(np.float32), 0, 255).astype(np.uint8)
        map_t1 = cv2.resize(map_t1, (200, 200))
        map_t2, box_info = render(traffic_meta.reshape(20, 20, 7), pixels_per_meter=20, t=2)
        map_t2 = map_t2[:400, 160:560]
        map_t2 = np.stack([map_t2, map_t2, map_t2], 2)
        map_t2 = np.clip(map_t2.astype(np.float32) + self_car_map.astype(np.float32), 0, 255).astype(np.uint8)
        map_t2 = cv2.resize(map_t2, (200, 200))


        if self.step % 2 != 0 and self.step > 4:
            control = self.prev_control
        else:
            self.prev_control = control
            self.prev_surround_map = surround_map

        meta_map = tick_data["raw"][:, 0].reshape(20, 20)  # 取第0通道
        meta_map = (meta_map - np.min(meta_map)) / (np.ptp(meta_map) + 1e-6)  # 归一化到0-1
        meta_map = (meta_map * 255).astype(np.uint8)
        meta_map = cv2.applyColorMap(cv2.resize(meta_map, (200, 200)), cv2.COLORMAP_JET)
        tick_data["meta_map"] = meta_map
        tick_data["map"] = self.prev_surround_map
        tick_data["map_t1"] = map_t1
        tick_data["map_t2"] = map_t2
        tick_data["rgb_raw"] = tick_data["rgb"]
        tick_data["rgb_left_raw"] = tick_data["rgb_left"]
        tick_data["rgb_right_raw"] = tick_data["rgb_right"]
        tick_data["rgb"] = cv2.resize(tick_data["rgb"], (800, 600))
        tick_data["rgb_left"] = cv2.resize(tick_data["rgb_left"], (200, 150))
        tick_data["rgb_right"] = cv2.resize(tick_data["rgb_right"], (200, 150))
        tick_data["rgb_focus"] = cv2.resize(tick_data["rgb_raw"][244:356, 344:456], (150, 150))
        tick_data["control"] = "throttle: %.2f, steer: %.2f, brake: %.2f" % (
            control.throttle,
            control.steer,
            control.brake,
        )
        tick_data["control_debug"] = {
            **control_debug,
            **lane_debug,
            "step": int(self.step),
            "speed": float(velocity),
            "target_x": float(tick_data["target_point"][0]),
            "target_y": float(tick_data["target_point"][1]),
            "raw_target_x": float(tick_data.get("target_point_raw", tick_data["target_point"])[0]),
            "raw_target_y": float(tick_data.get("target_point_raw", tick_data["target_point"])[1]),
            "net_is_junction": float(is_junction),
            "traffic_light_state": float(traffic_light_state),
            "stop_sign": float(stop_sign),
            "aux_junction": float(new_junction_prediction),
            "pred0_x": float(pred_waypoints[0, 0]),
            "pred0_y": float(pred_waypoints[0, 1]),
            "pred1_x": float(pred_waypoints[1, 0]),
            "pred1_y": float(pred_waypoints[1, 1]),
            "ctrl0_x": float(control_waypoints[0, 0]),
            "ctrl0_y": float(control_waypoints[0, 1]),
            "ctrl1_x": float(control_waypoints[1, 0]),
            "ctrl1_y": float(control_waypoints[1, 1]),
            "throttle": float(control.throttle),
            "steer": float(control.steer),
            "brake": float(control.brake),
        }
        tick_data["meta_infos"] = meta_infos
        tick_data["junction_prob"] = f"Junction Prob: {new_junction_prediction:.3f}"
        tick_data["box_info"] = "car: %d, bike: %d, pedestrian: %d" % (
            box_info["car"],
            box_info["bike"],
            box_info["pedestrian"],
        )
        tick_data["mes"] = "speed: %.2f" % velocity
        tick_data["time"] = "time: %.3f" % timestamp
        surface = self._hic.run_interface(tick_data)
        tick_data["surface"] = surface
       
        if SAVE_PATH is not None:
            self.save(tick_data)

        return control

    def save(self, tick_data):
        frame = self.step // self.skip_frames
        Image.fromarray(tick_data["surface"]).save(
            self.save_path / "meta" / ("%04d.jpg" % frame)
        )
        debug_path = self.save_path / "control.csv"
        debug = tick_data.get("control_debug", {})
        if debug:
            fields = [
                "step",
                "speed",
                "target_x",
                "target_y",
                "raw_target_x",
                "raw_target_y",
                "net_is_junction",
                "traffic_light_state",
                "stop_sign",
                "aux_junction",
                "raw_junction",
                "red_light_junction",
                "pred0_x",
                "pred0_y",
                "pred1_x",
                "pred1_y",
                "ctrl0_x",
                "ctrl0_y",
                "ctrl1_x",
                "ctrl1_y",
                "aim_x",
                "aim_y",
                "angle",
                "desired_speed",
                "safe_dis",
                "d_0",
                "d_05",
                "d_1",
                "stop_steps",
                "in_stop_sign_effect",
                "throttle",
                "steer",
                "brake",
                "lane_offset",
                "lane_width",
                "road_id",
                "lane_id",
                "steer_before_lane_center",
                "lane_center_correction",
            ]
            if not debug_path.exists():
                with open(debug_path, "w") as f:
                    f.write(",".join(fields) + "\n")
            with open(debug_path, "a") as f:
                f.write(",".join(str(debug.get(field, "")) for field in fields) + "\n")
    # 保存三张原始图片到 meta 同级别的文件夹
        # (self.save_path / "rgb_front").mkdir(parents=True, exist_ok=True)
        # (self.save_path / "rgb_left").mkdir(parents=True, exist_ok=True)
        # (self.save_path / "rgb_right").mkdir(parents=True, exist_ok=True)

        # Image.fromarray(tick_data["rgb_raw"]).save(
        #     self.save_path / "rgb_front" / ("%04d.jpg" % frame)
        # )
        # Image.fromarray(tick_data["rgb_left_raw"]).save(
        #     self.save_path / "rgb_left" / ("%04d.jpg" % frame)
        # )
        # Image.fromarray(tick_data["rgb_right_raw"]).save(
        #     self.save_path / "rgb_right" / ("%04d.jpg" % frame)
        # )
        return

    def destroy(self):
        if self.ensemble:
            del self.nets
        else:
            del self.net
