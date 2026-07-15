#!/usr/bin/env python
"""
Interfuser Data Collector using CARLA's BasicAgent
Collects all required data types for Interfuser training
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
try:
    from agents.navigation.behavior_agent import BehaviorAgent
    BEHAVIOR_AGENT_AVAILABLE = True
except ImportError:
    BEHAVIOR_AGENT_AVAILABLE = False
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from leaderboard.autoagents.autonomous_agent import AutonomousAgent, Track
from traffic_element_labels import (
    collect_traffic_element_labels,
    merge_legacy_affordances,
)


def get_entry_point():
    return 'InterfuserDataCollector'


class InterfuserDataCollector(AutonomousAgent):
    """
    Collects all Interfuser required data types using CARLA's BasicAgent for control
    """

    def setup(self, path_to_conf_file):
        """Setup the agent"""
        self.track = Track.SENSORS
        self._agent = None
        self._route_assigned = False
        
        # Data saving setup
        self.step = 0
        self.save_freq = 10  # Save at 2Hz (CARLA runs at 20Hz)
        self.route_counter = 0  # Track route number
        
        # Save base path for creating per-route directories
        self.base_save_path = os.environ.get('SAVE_PATH', None)
        self.save_path = None
        
        # Define data directories needed
        self.data_dirs = [
            # RGB cameras
            'rgb_front', 'rgb_left', 'rgb_right', 'rgb_rear',
            # Segmentation cameras
            'seg_front', 'seg_left', 'seg_right',
            # Depth cameras
            'depth_front', 'depth_left', 'depth_right',
            # LIDAR
            'lidar',
            # Birdview
            'birdview',
            # Bounding boxes
            '2d_bbs_front', '2d_bbs_left', '2d_bbs_right', '2d_bbs_rear',
            '3d_bbs',
            # Affordances and metadata
            'affordances',
            'traffic_elements',
            'measurements',
            'other_actors'
        ]
        
        print(f"[InterfuserDataCollector] Base path: {self.base_save_path}")

    def sensors(self):
        """Define all required sensors"""
        sensors = [
            # RGB cameras at 400x300 resolution
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
    
    def set_global_plan(self, global_plan_gps, global_plan_world_coord):
        """Called at the beginning of each new route"""
        super().set_global_plan(global_plan_gps, global_plan_world_coord)
        
        # Create a new directory for this route
        if self.base_save_path:
            self.route_counter += 1
            now = datetime.datetime.now()
            route_name = pathlib.Path(os.environ.get("ROUTES", "route")).stem
            string = f"{route_name}_route{self.route_counter:02d}_"
            string += "_".join(map(lambda x: "%02d" % x, (now.month, now.day, now.hour, now.minute, now.second)))
            
            self.save_path = pathlib.Path(self.base_save_path) / string
            self.save_path.mkdir(parents=True, exist_ok=True)
            
            # Create all data subdirectories
            for sensor_type in self.data_dirs:
                (self.save_path / sensor_type).mkdir(exist_ok=True)
            
            # Reset counters for new route
            self.step = 0
            self._agent = None
            self._route_assigned = False
            
            print(f"[InterfuserDataCollector] New route #{self.route_counter} - Data will be saved to: {self.save_path}")

    def run_step(self, input_data, timestamp):
        """Execute one step"""
        
        # Initialize Agent on first step
        if self._agent is None:
            hero_actor = CarlaDataProvider.get_hero_actor()
            if hero_actor:
                # Use BehaviorAgent for better traffic light handling
                if BEHAVIOR_AGENT_AVAILABLE:
                    self._agent = BehaviorAgent(
                        hero_actor,
                        behavior='normal',
                        opt_dict={
                            'base_tlight_threshold': 3.0,    # 红绿灯停车距离: 3米 (更接近!)
                            'base_vehicle_threshold': 8.0,   # 车辆避障距离: 8米
                            'base_min_distance': 3.0,        # 最小跟车距离: 3米
                            'max_brake': 0.8,                # 增加刹车力度以便更近距离停车
                        }
                    )
                    print(f"[InterfuserDataCollector] Using BehaviorAgent with tlight_threshold=3.0m")
                else:
                    self._agent = BasicAgent(hero_actor, target_speed=20)
                    print(f"[InterfuserDataCollector] Using BasicAgent (BehaviorAgent not available)")
                
                if hasattr(self, '_global_plan_world_coord') and self._global_plan_world_coord:
                    destination = self._global_plan_world_coord[-1][0].location
                    self._agent.set_destination(destination)
                    self._route_assigned = True
                    self._just_initialized = True  # 标记刚刚初始化，需要跳过第一帧
                    print(f"[InterfuserDataCollector] Route assigned, destination: {destination}")

        # Get control from agent
        # Skip first frame after initialization to let local planner initialize
        if self._agent and hasattr(self, '_just_initialized') and self._just_initialized:
            # First frame after set_destination - let local planner initialize
            self._just_initialized = False
            control = carla.VehicleControl()
            control.throttle = 0.3
            control.steer = 0.0
            control.brake = 0.0
        elif self._agent:
            try:
                control = self._agent.run_step()
            except AttributeError as e:
                # BehaviorAgent may have None _incoming_waypoint in early frames
                # while local planner initializes - use simple forward control
                if "'NoneType' object has no attribute 'is_junction'" in str(e):
                    control = carla.VehicleControl()
                    control.throttle = 0.3
                    control.steer = 0.0
                    control.brake = 0.0
                else:
                    raise  # Re-raise other AttributeErrors
        else:
            control = carla.VehicleControl()
            control.steer = 0.0
            control.throttle = 0.0
            control.brake = 1.0

        # Save data at specified frequency
        if self.save_path and self.step % self.save_freq == 0:
            self._save_all_data(input_data, control, timestamp)
        
        self.step += 1
        return control

    def _save_all_data(self, input_data, control, timestamp):
        """Save all required data types"""
        frame = self.step // self.save_freq

        hero = CarlaDataProvider.get_hero_actor()
        world = CarlaDataProvider.get_world()
        if not hero or not world:
            return
        traffic_elements = collect_traffic_element_labels(hero, world)

        # 1. RGB images (400x300)
        for cam in ['rgb_front', 'rgb_left', 'rgb_right', 'rgb_rear']:
            if cam in input_data:
                img = cv2.cvtColor(input_data[cam][1][:, :, :3], cv2.COLOR_BGR2RGB)
                Image.fromarray(img).save(self.save_path / cam / f"{frame:04d}.jpg")
        
        # 2. Segmentation images
        for cam in ['seg_front', 'seg_left', 'seg_right']:
            if cam in input_data:
                seg = input_data[cam][1][:, :, 2]  # Red channel contains semantic tags
                Image.fromarray(seg).save(self.save_path / cam / f"{frame:04d}.png")
        
        # 3. Depth images
        for cam in ['depth_front', 'depth_left', 'depth_right']:
            if cam in input_data:
                depth = input_data[cam][1][:, :, :3]
                cv2.imwrite(str(self.save_path / cam / f"{frame:04d}.png"), depth)
        
        # 4. LIDAR point cloud
        if 'lidar' in input_data:
            lidar_data = input_data['lidar'][1]
            np.save(self.save_path / 'lidar' / f"{frame:04d}.npy", lidar_data)
        
        with open(self.save_path / 'traffic_elements' / f"{frame:04d}.json", 'w') as f:
            json.dump(traffic_elements, f, indent=2)
        
        # 5. Birdview (top-down segmentation)
        birdview = self._generate_birdview(hero, world)
        Image.fromarray(birdview).save(self.save_path / 'birdview' / f"{frame:04d}.png")
        
        # 6. 3D bounding boxes
        bb_3d = self._get_3d_bounding_boxes(hero, world)
        with open(self.save_path / '3d_bbs' / f"{frame:04d}.json", 'w') as f:
            json.dump(bb_3d, f, indent=2)
        
        # 7. 2D bounding boxes (projected from 3D)
        for cam in ['front', 'left', 'right', 'rear']:
            bb_2d = self._get_2d_bounding_boxes(bb_3d, cam, input_data)
            with open(self.save_path / f'2d_bbs_{cam}' / f"{frame:04d}.json", 'w') as f:
                json.dump(bb_2d, f, indent=2)
        
        # 8. Affordances
        affordances = self._get_affordances(hero, world, traffic_elements)
        with open(self.save_path / 'affordances' / f"{frame:04d}.json", 'w') as f:
            json.dump(affordances, f, indent=2)
        
        # 9. Measurements (ego vehicle metadata)
        measurements = self._get_measurements(hero, input_data, control, timestamp)
        with open(self.save_path / 'measurements' / f"{frame:04d}.json", 'w') as f:
            json.dump(measurements, f, indent=2)
        
        # 10. Other actors (vehicles and traffic lights)
        other_actors = self._get_other_actors(hero, world)
        with open(self.save_path / 'other_actors' / f"{frame:04d}.json", 'w') as f:
            json.dump(other_actors, f, indent=2)

    def _generate_birdview(self, hero, world):
        """Generate top-down segmentation view"""
        # Simple birdview: render a 192x192 top-down view
        # For full implementation, this would render the map and vehicles
        size = 192
        pixels_per_meter = 5
        
        birdview = np.zeros((size, size, 3), dtype=np.uint8)
        birdview[:, :] = [128, 128, 128]  # Gray background (road)
        
        # Get hero position
        hero_transform = hero.get_transform()
        hero_loc = hero_transform.location
        
        # Draw nearby vehicles
        vehicles = world.get_actors().filter('vehicle.*')
        for vehicle in vehicles:
            if vehicle.id == hero.id:
                continue
            
            v_loc = vehicle.get_location()
            dx = v_loc.x - hero_loc.x
            dy = v_loc.y - hero_loc.y
            
            # Convert to birdview coordinates
            px = int(size / 2 + dy * pixels_per_meter)
            py = int(size / 2 - dx * pixels_per_meter)
            
            if 0 <= px < size and 0 <= py < size:
                cv2.circle(birdview, (px, py), 3, (255, 0, 0), -1)  # Blue for vehicles
        
        # Draw ego vehicle at center
        cv2.circle(birdview, (size // 2, size // 2), 4, (0, 255, 0), -1)  # Green for ego
        
        return birdview

    def _get_3d_bounding_boxes(self, hero, world):
        """Get 3D bounding boxes for all nearby actors"""
        bounding_boxes = []
        hero_loc = hero.get_location()
        
        # Get all vehicles
        vehicles = world.get_actors().filter('vehicle.*')
        for vehicle in vehicles:
            if vehicle.id == hero.id:
                continue
            
            v_loc = vehicle.get_location()
            distance = hero_loc.distance(v_loc)
            
            if distance < 50:  # Only within 50 meters
                bb = vehicle.bounding_box
                bounding_boxes.append({
                    'type': 'vehicle',
                    'id': vehicle.id,
                    'location': {'x': v_loc.x, 'y': v_loc.y, 'z': v_loc.z},
                    'extent': {'x': bb.extent.x, 'y': bb.extent.y, 'z': bb.extent.z},
                    'rotation': {
                        'pitch': vehicle.get_transform().rotation.pitch,
                        'yaw': vehicle.get_transform().rotation.yaw,
                        'roll': vehicle.get_transform().rotation.roll
                    }
                })
        
        # Get traffic lights
        traffic_lights = world.get_actors().filter('traffic.traffic_light*')
        for tl in traffic_lights:
            tl_loc = tl.get_location()
            distance = hero_loc.distance(tl_loc)
            
            if distance < 50:
                bounding_boxes.append({
                    'type': 'traffic_light',
                    'id': tl.id,
                    'location': {'x': tl_loc.x, 'y': tl_loc.y, 'z': tl_loc.z},
                    'state': str(tl.state)
                })
        
        return bounding_boxes

    def _get_2d_bounding_boxes(self, bb_3d, camera, input_data):
        """Project 3D bounding boxes to 2D camera view"""
        # Simplified: return bounding box centers projected to 2D
        # Full implementation would project all 8 corners of the 3D box
        bb_2d = []
        
        for bb in bb_3d:
            if bb['type'] == 'vehicle':
                # Simple projection (would need proper camera matrix for full implementation)
                bb_2d.append({
                    'type': bb['type'],
                    'id': bb['id'],
                    # Placeholder - proper projection requires camera intrinsics
                    'bbox': [0, 0, 0, 0]  # [x_min, y_min, x_max, y_max]
                })
        
        return bb_2d

    def _get_affordances(self, hero, world, traffic_elements=None):
        """Get affordances: traffic lights, stop signs, hazard detection"""
        affordances = {
            'hazard_vehicle': False,
            'hazard_pedestrian': False
        }

        if traffic_elements is None:
            traffic_elements = collect_traffic_element_labels(hero, world)
        
        # Check for nearby hazards
        hero_loc = hero.get_location()
        vehicles = world.get_actors().filter('vehicle.*')
        
        for vehicle in vehicles:
            if vehicle.id == hero.id:
                continue
            
            distance = hero_loc.distance(vehicle.get_location())
            if distance < 10:  # Hazard within 10 meters
                affordances['hazard_vehicle'] = True
                break
        
        return merge_legacy_affordances(affordances, traffic_elements)

    def _get_measurements(self, hero, input_data, control, timestamp):
        """Get ego vehicle measurements"""
        loc = hero.get_location()
        vel = hero.get_velocity()
        acc = hero.get_acceleration()
        
        gps = input_data['gps'][1][:2] if 'gps' in input_data else [0, 0]
        speed = input_data['speed'][1]['speed'] if 'speed' in input_data else 0
        compass = input_data['imu'][1][-1] if 'imu' in input_data else 0
        
        measurements = {
            'x': loc.x,
            'y': loc.y,
            'z': loc.z,
            'pitch': hero.get_transform().rotation.pitch,
            'yaw': hero.get_transform().rotation.yaw,
            'roll': hero.get_transform().rotation.roll,
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
        
        # Add route command
        if self._agent and hasattr(self._agent, 'get_local_planner'):
            local_planner = self._agent.get_local_planner()
            if local_planner:
                try:
                    # Pass debug=False to avoid returning visualization data
                    waypoint, direction = local_planner.get_incoming_waypoint_and_direction(steps=3)
                    if waypoint:
                        measurements['x_command'] = waypoint.transform.location.x
                        measurements['y_command'] = waypoint.transform.location.y
                        measurements['command'] = direction.value
                except Exception as e:
                    print(f"Error getting waypoint command: {e}")
                    measurements['command'] = 4  # LANE_FOLLOW fallback
        
        return measurements

    def _get_other_actors(self, hero, world):
        """Get positions and metadata of surrounding vehicles and traffic lights"""
        actors_data = []
        hero_loc = hero.get_location()
        
        # Get vehicles
        vehicles = world.get_actors().filter('vehicle.*')
        for vehicle in vehicles:
            if vehicle.id == hero.id:
                continue
            
            v_loc = vehicle.get_location()
            distance = hero_loc.distance(v_loc)
            
            if distance < 50:
                v_vel = vehicle.get_velocity()
                actors_data.append({
                    'type': 'vehicle',
                    'id': vehicle.id,
                    'type_id': vehicle.type_id,
                    'x': v_loc.x,
                    'y': v_loc.y,
                    'z': v_loc.z,
                    'yaw': vehicle.get_transform().rotation.yaw,
                    'velocity_x': v_vel.x,
                    'velocity_y': v_vel.y,
                    'velocity_z': v_vel.z,
                })
        
        # Get traffic lights
        traffic_lights = world.get_actors().filter('traffic.traffic_light*')
        for tl in traffic_lights:
            tl_loc = tl.get_location()
            distance = hero_loc.distance(tl_loc)
            
            if distance < 50:
                actors_data.append({
                    'type': 'traffic_light',
                    'id': tl.id,
                    'x': tl_loc.x,
                    'y': tl_loc.y,
                    'z': tl_loc.z,
                    'state': str(tl.state)
                })
        
        return actors_data

    def destroy(self):
        """Cleanup"""
        if self._agent:
            del self._agent
