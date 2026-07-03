#!/usr/bin/env python
"""
Data Collection Agent using CARLA's BehaviorAgent
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

from agents.navigation.basic_agent import BasicAgent
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from leaderboard.autoagents.autonomous_agent import AutonomousAgent, Track


def get_entry_point():
    return 'BehaviorAgentCollector'


class BehaviorAgentCollector(AutonomousAgent):
    """
    Uses CARLA's BasicAgent for control and saves training data
    """

    def setup(self, path_to_conf_file):
        """Setup the agent"""
        self.track = Track.SENSORS
        self._agent = None
        self._route_assigned = False
        
        # Data saving setup
        self.step = 0
        self.save_freq = 10  # Save at 2Hz (CARLA runs at 20Hz, so save every 10 frames)
        
        save_path = os.environ.get('SAVE_PATH', None)
        if save_path:
            now = datetime.datetime.now()
            string = pathlib.Path(os.environ.get("ROUTES", "route")).stem + "_"
            string += "_".join(map(lambda x: "%02d" % x, (now.month, now.day, now.hour, now.minute, now.second)))
            
            self.save_path = pathlib.Path(save_path) / string
            self.save_path.mkdir(parents=True, exist_ok=True)
            
            # Create directories for data
            for sensor_type in ['rgb_front', 'rgb_left', 'rgb_right', 'rgb_rear', 
                               'lidar', 'measurements', 'topdown', 'birdview']:
                (self.save_path / sensor_type).mkdir(exist_ok=True)
            
            print(f"Data will be saved to: {self.save_path}")
        else:
            self.save_path = None

    def sensors(self):
        """Define required sensors"""
        sensors = [
            # RGB cameras
            {'type': 'sensor.camera.rgb', 'x': 1.3, 'y': 0.0, 'z': 2.3, 
             'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
             'width': 800, 'height': 600, 'fov': 100, 'id': 'rgb_front'},
            
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
        
        return sensors

    def run_step(self, input_data, timestamp):
        """Execute one step"""
        
        # Initialize BasicAgent on first step
        if self._agent is None:
            hero_actor = CarlaDataProvider.get_hero_actor()
            if hero_actor:
                # Create BasicAgent with target speed (more stable than BehaviorAgent)
                self._agent = BasicAgent(hero_actor, target_speed=20)
                
                # Set destination from global plan
                if hasattr(self, '_global_plan_world_coord') and self._global_plan_world_coord:
                    # Get the final destination
                    destination = self._global_plan_world_coord[-1][0].location
                    self._agent.set_destination(destination)
                    self._route_assigned = True
                    print(f"Route assigned to BasicAgent, destination: {destination}")

        # Get control from BehaviorAgent
        if self._agent:
            control = self._agent.run_step()
        else:
            # Fallback: stop the vehicle
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
        """Save sensor data and measurements"""
        frame = self.step // self.save_freq
        
        # Save RGB images
        for cam in ['rgb_front', 'rgb_left', 'rgb_right', 'rgb_rear']:
            if cam in input_data:
                img = cv2.cvtColor(input_data[cam][1][:, :, :3], cv2.COLOR_BGR2RGB)
                Image.fromarray(img).save(self.save_path / cam / f"{frame:04d}.jpg")
        
        # Save LIDAR
        if 'lidar' in input_data:
            lidar_data = input_data['lidar'][1]
            np.save(self.save_path / 'lidar' / f"{frame:04d}.npy", lidar_data)
        
        # Save measurements
        if 'gps' in input_data and 'imu' in input_data and 'speed' in input_data:
            gps = input_data['gps'][1][:2]
            speed = input_data['speed'][1]['speed']
            compass = input_data['imu'][1][-1]
            
            # Get vehicle location from CARLA
            hero = CarlaDataProvider.get_hero_actor()
            if hero:
                loc = hero.get_location()
                
                # Get world reference
                world = CarlaDataProvider.get_world()
                world_map = world.get_map()
                
                measurements = {
                    'x': loc.x,
                    'y': loc.y,
                    'z': loc.z,
                    'gps_x': gps[1] * 111324.60662786,  # Simple conversion
                    'gps_y': gps[0] * 111324.60662786,
                    'theta': compass,
                    'speed': speed,
                    'steer': control.steer,
                    'throttle': control.throttle,
                    'brake': control.brake,
                    'timestamp': timestamp,
                }
                
                # Add route information if available
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
                            pass
                
                # Save JSON
                with open(self.save_path / 'measurements' / f"{frame:04d}.json", 'w') as f:
                    json.dump(measurements, f, indent=4)

    def destroy(self):
        """Cleanup"""
        if self._agent:
            del self._agent

