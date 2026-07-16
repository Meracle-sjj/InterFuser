#!/usr/bin/env python
"""
Complete Interfuser Data Collector with ALL detection logic from base_agent
Compatible with CARLA 0.9.16
Includes: is_junction, is_vehicle_present, is_red_light_present, etc.
"""
import sys
import os
import numpy as np
import carla

# Import base collector
sys.path.insert(0, os.path.dirname(__file__))
try:
    from .interfuser_data_collector import InterfuserDataCollector as BaseCollector
except ImportError:
    from interfuser_data_collector import InterfuserDataCollector as BaseCollector


def get_entry_point():
    return 'InterfuserCollectorComplete'


def _numpy(carla_vector):
    """Convert carla.Vector3D to numpy array"""
    return np.array([carla_vector.x, carla_vector.y, carla_vector.z])


def _orientation(yaw):
    """Convert yaw angle to orientation vector"""
    return np.array([np.cos(np.radians(yaw)), np.sin(np.radians(yaw)), 0])


class InterfuserCollectorComplete(BaseCollector):
    """
    Complete data collector with all detection logic from auto_pilot.py
    """
    
    def _get_measurements(self, hero, input_data, control, timestamp):
        """Enhanced measurements with REAL detection from CARLA"""
        # Get base measurements
        measurements = super()._get_measurements(hero, input_data, control, timestamp)
        
        # Get world and map
        world = hero.get_world()
        world_map = world.get_map()
        hero_loc = hero.get_location()
        
        # ========== 1. is_junction ==========
        waypoint = world_map.get_waypoint(hero_loc)
        measurements['is_junction'] = waypoint.is_junction
        
        # ========== 2. Get all actors ==========
        actors = world.get_actors()
        vehicles = actors.filter('vehicle.*')
        walkers = actors.filter('walker.*')
        traffic_lights = actors.filter('traffic.traffic_light*')
        
        # ========== 3. Vehicle detection ==========
        is_vehicle_present = []
        is_bike_present = []
        is_lane_vehicle_present = []
        is_junction_vehicle_present = []
        
        hero_transform = hero.get_transform()
        hero_forward = hero_transform.get_forward_vector()
        hero_yaw = hero_transform.rotation.yaw
        
        for vehicle in vehicles:
            if vehicle.id == hero.id or not vehicle.is_alive:
                continue
            
            v_loc = vehicle.get_location()
            distance = hero_loc.distance(v_loc)
            
            if distance > 50:  # Too far
                continue
            
            # Check if it's in front of us
            to_vehicle = v_loc - hero_loc
            forward_dot = (to_vehicle.x * hero_forward.x + 
                          to_vehicle.y * hero_forward.y)
            
            if forward_dot > 0 and distance < 30:  # In front and close
                is_vehicle_present.append(vehicle.id)
                
                # Check if it's a bike/motorcycle
                if 'bike' in vehicle.type_id.lower() or 'motor' in vehicle.type_id.lower():
                    is_bike_present.append(vehicle.id)
                
                # Check if in junction
                v_waypoint = world_map.get_waypoint(v_loc)
                if v_waypoint and v_waypoint.is_junction:
                    is_junction_vehicle_present.append(vehicle.id)
                
                # Check if in same lane (closer vehicles)
                if distance < 15:
                    # Check if waypoint road_id and lane_id match
                    if v_waypoint and waypoint:
                        if (v_waypoint.road_id == waypoint.road_id and 
                            v_waypoint.lane_id == waypoint.lane_id):
                            is_lane_vehicle_present.append(vehicle.id)
        
        measurements['is_vehicle_present'] = is_vehicle_present
        measurements['is_bike_present'] = is_bike_present
        measurements['is_lane_vehicle_present'] = is_lane_vehicle_present
        measurements['is_junction_vehicle_present'] = is_junction_vehicle_present
        
        # ========== 4. Pedestrian detection ==========
        is_pedestrian_present = []
        
        for walker in walkers:
            w_loc = walker.get_location()
            distance = hero_loc.distance(w_loc)
            
            if distance < 30:  # Within 30 meters
                # Check if in front
                to_walker = w_loc - hero_loc
                forward_dot = (to_walker.x * hero_forward.x + 
                              to_walker.y * hero_forward.y)
                
                if forward_dot > 0:  # In front
                    is_pedestrian_present.append(walker.id)
        
        measurements['is_pedestrian_present'] = is_pedestrian_present
        
        # ========== 5. Traffic light detection ==========
        is_red_light_present = []
        affected_light_id = -1
        
        # Method 1: Check if vehicle is affected by traffic light
        traffic_light = hero.get_traffic_light()
        if traffic_light is not None:
            tl_state = traffic_light.state
            if tl_state == carla.TrafficLightState.Red or tl_state == carla.TrafficLightState.Yellow:
                is_red_light_present.append(traffic_light.id)
                affected_light_id = traffic_light.id
        
        # Method 2: Find nearby red lights (backup method)
        if len(is_red_light_present) == 0:
            for tl in traffic_lights:
                tl_loc = tl.get_location()
                distance = hero_loc.distance(tl_loc)
                
                if distance < 20 and tl.state == carla.TrafficLightState.Red:
                    # Check if in front
                    to_tl = tl_loc - hero_loc
                    forward_dot = (to_tl.x * hero_forward.x + 
                                  to_tl.y * hero_forward.y)
                    
                    if forward_dot > 0:
                        is_red_light_present.append(tl.id)
                        if affected_light_id == -1:
                            affected_light_id = tl.id
                        break  # Only first one
        
        measurements['is_red_light_present'] = is_red_light_present
        measurements['affected_light_id'] = affected_light_id
        
        # ========== 6. Behavior hints ==========
        should_brake = 0
        should_slow = 0
        
        # Should brake if:
        # - Red light present
        # - Vehicle very close in lane
        # - Pedestrian crossing
        if len(is_red_light_present) > 0:
            should_brake = 1
        elif len(is_lane_vehicle_present) > 0:
            should_brake = 1
        elif len(is_pedestrian_present) > 3:  # Many pedestrians
            should_brake = 1
        
        # Should slow if vehicles nearby but not immediately dangerous
        if not should_brake:
            if len(is_vehicle_present) > 2:
                should_slow = 1
            elif len(is_junction_vehicle_present) > 0:
                should_slow = 1
        
        measurements['should_brake'] = should_brake
        measurements['should_slow'] = should_slow
        
        # ========== 7. Future waypoints ==========
        future_waypoints = []
        try:
            current_waypoint = waypoint
            for i in range(50):
                next_wps = current_waypoint.next(2.0)  # 2 meters ahead
                if next_wps:
                    current_waypoint = next_wps[0]
                    loc = current_waypoint.transform.location
                    future_waypoints.append([loc.x, loc.y, loc.z])
                else:
                    break
        except:
            pass
        
        measurements['future_waypoints'] = future_waypoints
        
        # ========== 8. Near/Far nodes (for compatibility) ==========
        if getattr(self, '_agent', None) and hasattr(self._agent, 'get_local_planner'):
            local_planner = self._agent.get_local_planner()
            if local_planner:
                try:
                    waypoint, direction = local_planner.get_incoming_waypoint_and_direction(steps=3)
                    if waypoint:
                        measurements['near_node_x'] = waypoint.transform.location.x
                        measurements['near_node_y'] = waypoint.transform.location.y
                        
                        # Far node (further ahead)
                        far_waypoint, _ = local_planner.get_incoming_waypoint_and_direction(steps=10)
                        if far_waypoint:
                            measurements['far_node_x'] = far_waypoint.transform.location.x
                            measurements['far_node_y'] = far_waypoint.transform.location.y
                except:
                    pass
        
        return measurements


if __name__ == '__main__':
    print("Complete Interfuser Data Collector - with REAL detection for CARLA 0.9.16")
    print("Includes: is_junction, vehicle/pedestrian/traffic light detection, etc.")
