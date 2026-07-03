#!/usr/bin/env python
"""
Complete Data Collection Agent using CARLA's BasicAgent
Collects all data types required by Interfuser
Compatible with CARLA 0.9.16
"""

import os
import json
import pathlib
import datetime
import numpy as np
import cv2
import carla
from PIL import Image
from collections import deque

from agents.navigation.basic_agent import BasicAgent
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from leaderboard.autoagents.autonomous_agent import AutonomousAgent, Track


def get_entry_point():
    return 'CompleteDataCollector'


class CompleteDataCollector(AutonomousAgent):
    """
    Uses CARLA's BasicAgent for control and saves all Interfuser training data
    """

    def setup(self, path_to_conf_file):
        """Setup the agent"""
        self.track = Track.SENSORS
        self._agent = None
        self._route_assigned = False
        
        # Data saving setup
        self.step = 0
        self.save_freq = 10  # Save at 2Hz (CARLA runs at 20Hz)
        
        # Read configuration
        self.config_path = path_to_conf_file
        self.rgb_only = True  # Set to False to enable depth/seg sensors
        
        save_path = os.environ.get('SAVE_PATH', None)
        if save_path:
            now = datetime.datetime.now()
            string = pathlib.Path(os.environ.get("ROUTES", "route")).stem + "_"
            string += "_".join(map(lambda x: "%02d" % x, (now.month, now.day, now.hour, now.minute, now.second)))
            
            self.save_path = pathlib.Path(save_path) / string
            self.save_path.mkdir(parents=True, exist_ok=True)
            
            # Create directories for all data types
            data_dirs = [
                'rgb_front', 'rgb_left', 'rgb_right', 'rgb_rear',
                'lidar', 'measurements', 'birdview',
                '2d_bbs_front', '2d_bbs_left', '2d_bbs_right', '2d_bbs_rear',
                '3d_bbs', 'affordances', 'other_actors'
            ]
            
            # Add depth and seg dirs if not rgb_only
            if not self.rgb_only:
                data_dirs.extend([
                    'seg_front', 'seg_left', 'seg_right',
                    'depth_front', 'depth_left', 'depth_right'
                ])
            
            for sensor_type in data_dirs:
                (self.save_path / sensor_type).mkdir(exist_ok=True)
            
            print(f"Data will be saved to: {self.save_path}")
            print(f"RGB only mode: {self.rgb_only}")
        else:
            self.save_path = None

    def sensors(self):
        """Define required sensors"""
        sensors = [
            # RGB cameras (400x300 resolution as per Interfuser spec)
            {'type': 'sensor.camera.rgb', 'x': 1.3, 'y': 0.0, 'z': 2.3, 
             'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
             'width': 400, 'height': 300, 'fov': 100, 'id': 'rgb_front'},
            
            {'type': 'sensor.camera.rgb', 'x': 1.3, 'y': 0.0, 'z': 2.3,
             'roll': 0.0, 'pitch': 0.0, 'yaw': -60.0,
             'width': 400, 'height': 300, 'fov': 100, 'id': 'rgb_left'},
            
            {'type': 'sensor.camera.rgb', 'x': 1.3, 'y': 0.0, 'z': 2.3,
             'roll': 0.0, 'pitch': 0.0, 'yaw': 60.0,
             'width': 400, 'height': 300, 'fov': 100, 'id': 'rgb_right'},
            
            {'type': 'sensor.camera.rgb', 'x': -1.3, 'y': 0.0, 'z': 2.3,
             'roll': 0.0, 'pitch': 0.0, 'yaw': 180.0,
             'width': 400, 'height': 300, 'fov': 100, 'id': 'rgb_rear'},
            
            # LIDAR
            {'type': 'sensor.lidar.ray_cast', 'x': 1.3, 'y': 0.0, 'z': 2.5,
             'roll': 0.0, 'pitch': 0.0, 'yaw': -90.0, 'id': 'lidar'},
            
            # GPS and IMU
            {'type': 'sensor.other.gnss', 'x': 0.0, 'y': 0.0, 'z': 0.0,
             'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0, 'sensor_tick': 0.01, 'id': 'gps'},
            
            {'type': 'sensor.other.imu', 'x': 0.0, 'y': 0.0, 'z': 0.0,
             'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0, 'sensor_tick': 0.05, 'id': 'imu'},
            
            # Speedometer
            {'type': 'sensor.speedometer', 'reading_frequency': 20, 'id': 'speed'},
        ]
        
        # Add depth and segmentation cameras if not rgb_only
        if not self.rgb_only:
            sensors.extend([
                # Segmentation cameras
                {'type': 'sensor.camera.semantic_segmentation', 'x': 1.3, 'y': 0.0, 'z': 2.3,
                 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                 'width': 400, 'height': 300, 'fov': 100, 'id': 'seg_front'},
                
                {'type': 'sensor.camera.semantic_segmentation', 'x': 1.3, 'y': 0.0, 'z': 2.3,
                 'roll': 0.0, 'pitch': 0.0, 'yaw': -60.0,
                 'width': 400, 'height': 300, 'fov': 100, 'id': 'seg_left'},
                
                {'type': 'sensor.camera.semantic_segmentation', 'x': 1.3, 'y': 0.0, 'z': 2.3,
                 'roll': 0.0, 'pitch': 0.0, 'yaw': 60.0,
                 'width': 400, 'height': 300, 'fov': 100, 'id': 'seg_right'},
                
                # Depth cameras
                {'type': 'sensor.camera.depth', 'x': 1.3, 'y': 0.0, 'z': 2.3,
                 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                 'width': 400, 'height': 300, 'fov': 100, 'id': 'depth_front'},
                
                {'type': 'sensor.camera.depth', 'x': 1.3, 'y': 0.0, 'z': 2.3,
                 'roll': 0.0, 'pitch': 0.0, 'yaw': -60.0,
                 'width': 400, 'height': 300, 'fov': 100, 'id': 'depth_left'},
                
                {'type': 'sensor.camera.depth', 'x': 1.3, 'y': 0.0, 'z': 2.3,
                 'roll': 0.0, 'pitch': 0.0, 'yaw': 60.0,
                 'width': 400, 'height': 300, 'fov': 100, 'id': 'depth_right'},
            ])
        
        return sensors

    def run_step(self, input_data, timestamp):
        """Execute one step"""
        
        # Initialize BasicAgent on first step
        if self._agent is None:
            hero_actor = CarlaDataProvider.get_hero_actor()
            if hero_actor:
                self._agent = BasicAgent(hero_actor, target_speed=20)
                
                if hasattr(self, '_global_plan_world_coord') and self._global_plan_world_coord:
                    destination = self._global_plan_world_coord[-1][0].location
                    self._agent.set_destination(destination)
                    self._route_assigned = True
                    print(f"Route assigned to BasicAgent, destination: {destination}")

        # Get control from BasicAgent
        if self._agent:
            control = self._agent.run_step()
        else:
            control = carla.VehicleControl()
            control.steer = 0.0
            control.throttle = 0.0
            control.brake = 1.0

        # Save data
        if self.save_path and self.step % self.save_freq == 0:
            self._save_data(input_data, control, timestamp)
        
        self.step += 1
        return control

    def _save_data(self, input_data, control, timestamp):
        """Save all sensor data and metadata"""
        frame = self.step // self.save_freq
        
        # 1. Save RGB images
        for cam in ['rgb_front', 'rgb_left', 'rgb_right', 'rgb_rear']:
            if cam in input_data:
                img = cv2.cvtColor(input_data[cam][1][:, :, :3], cv2.COLOR_BGR2RGB)
                Image.fromarray(img).save(self.save_path / cam / f"{frame:04d}.jpg")
        
        # 2. Save depth and segmentation (if available)
        if not self.rgb_only:
            for cam in ['seg_front', 'seg_left', 'seg_right']:
                if cam in input_data:
                    seg = input_data[cam][1][:, :, 2]  # Red channel contains semantic tags
                    Image.fromarray(seg).save(self.save_path / cam / f"{frame:04d}.png")
            
            for cam in ['depth_front', 'depth_left', 'depth_right']:
                if cam in input_data:
                    depth = input_data[cam][1][:, :, :3]
                    cv2.imwrite(str(self.save_path / cam / f"{frame:04d}.png"), depth)
        
        # 3. Save LIDAR
        if 'lidar' in input_data:
            lidar_data = input_data['lidar'][1]
            np.save(self.save_path / 'lidar' / f"{frame:04d}.npy", lidar_data)
        
        # 4. Get world and actors info
        hero = CarlaDataProvider.get_hero_actor()
        world = CarlaDataProvider.get_world()
        
        if hero and world:
            # Get ego vehicle data
            loc = hero.get_location()
            vel = hero.get_velocity()
            acc = hero.get_acceleration()
            
            gps = input_data['gps'][1][:2] if 'gps' in input_data else [0, 0]
            speed = input_data['speed'][1]['speed'] if 'speed' in input_data else 0
            compass = input_data['imu'][1][-1] if 'imu' in input_data else 0
            
            # 5. Save measurements
            measurements = {
                'x': loc.x,
                'y': loc.y,
                'z': loc.z,
                'gps_x': gps[1] * 111324.60662786,
                'gps_y': gps[0] * 111324.60662786,
                'theta': compass,
                'speed': speed,
                'velocity_x': vel.x,
                'velocity_y': vel.y,
                'velocity_z': vel.z,
                'acceleration_x': acc.x,
                'acceleration_y': acc.y,
                'acceleration_z': acc.z,
                'steer': control.steer,
                'throttle': control.throttle,
                'brake': control.brake,
                'hand_brake': control.hand_brake,
                'reverse': control.reverse,
                'timestamp': timestamp,
            }
            
            # Add route command if available
            if self._agent and hasattr(self._agent, 'get_local_planner'):
                local_planner = self._agent.get_local_planner()
                if local_planner and hasattr(local_planner, 'get_incoming_waypoint_and_direction'):
                    try:
                        waypoint, direction = local_planner.get_incoming_waypoint_and_direction(steps=3)
                        if waypoint:
                            measurements['x_command'] = waypoint.transform.location.x
                            measurements['y_command'] = waypoint.transform.location.y
                            measurements['command'] = direction.value
                    except:
                        measurements['command'] = 4  # LANE_FOLLOW as default
            
            with open(self.save_path / 'measurements' / f"{frame:04d}.json", 'w') as f:
                json.dump(measurements, f, indent=4)
            
            # 6. Save other actors info
            actors_data = []
            all_actors = world.get_actors()
            vehicles = all_actors.filter('vehicle.*')
            
            for vehicle in vehicles:
                if vehicle.id != hero.id:  # Skip ego vehicle
                    v_loc = vehicle.get_location()
                    v_vel = vehicle.get_velocity()
                    
                    actors_data.append({
                        'id': vehicle.id,
                        'type': vehicle.type_id,
                        'x': v_loc.x,
                        'y': v_loc.y,
                        'z': v_loc.z,
                        'velocity_x': v_vel.x,
                        'velocity_y': v_vel.y,
                        'velocity_z': v_vel.z,
                    })
            
            with open(self.save_path / 'other_actors' / f"{frame:04d}.json", 'w') as f:
                json.dump(actors_data, f, indent=4)
            
            # 7. Save bounding boxes (simplified - for full implementation, refer to base_agent.py)
            # For now, save empty placeholders to maintain directory structure
            for bb_dir in ['2d_bbs_front', '2d_bbs_left', '2d_bbs_right', '2d_bbs_rear', '3d_bbs']:
                with open(self.save_path / bb_dir / f"{frame:04d}.json", 'w') as f:
                    json.dump([], f)
            
            # 8. Save affordances (placeholder)
            with open(self.save_path / 'affordances' / f"{frame:04d}.json", 'w') as f:
                json.dump({}, f)
            
            # 9. Save birdview (placeholder - requires rendering)
            # Create empty birdview for now
            birdview = np.zeros((192, 192, 3), dtype=np.uint8)
            Image.fromarray(birdview).save(self.save_path / 'birdview' / f"{frame:04d}.png")

    def destroy(self):
        """Cleanup"""
        if self._agent:
            del self._agent

