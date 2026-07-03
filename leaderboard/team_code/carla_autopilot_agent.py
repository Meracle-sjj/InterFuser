#!/usr/bin/env python

"""
CARLA Autopilot Agent for Data Collection
Uses CARLA's built-in autopilot with data saving functionality
Compatible with CARLA 0.9.16
"""

import carla
from agents.navigation.basic_agent import BasicAgent
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

from leaderboard.autoagents.autonomous_agent import AutonomousAgent, Track
from team_code.base_agent import BaseAgent


def get_entry_point():
    return 'CarlaAutopilotAgent'


class CarlaAutopilotAgent(BaseAgent):
    """
    Agent that uses CARLA's built-in autopilot for control
    but inherits data saving functionality from BaseAgent
    """

    def setup(self, path_to_conf_file):
        """Setup the agent parameters"""
        # Call parent setup for data saving configuration
        super().setup(path_to_conf_file)
        
        self.track = Track.SENSORS
        self._agent = None
        self._route_assigned = False
        
        # Initialize CARLA world access
        self._world = CarlaDataProvider.get_world()
        
        # Initialize step counter if not set by parent
        if not hasattr(self, 'step'):
            self.step = 0
        if not hasattr(self, 'save_freq'):
            self.save_freq = 5  # Default save frequency

    def sensors(self):
        """Use BaseAgent's sensor configuration"""
        return super().sensors()

    def run_step(self, input_data, timestamp):
        """
        Execute one step of navigation using CARLA autopilot
        """
        # Initialize CARLA autopilot if not done yet
        if not self._agent:
            hero_actor = CarlaDataProvider.get_hero_actor()
            if hero_actor:
                self._agent = BasicAgent(hero_actor, target_speed=20)
                
                # Set route if available
                if self._global_plan_world_coord:
                    plan = []
                    for transform, cmd in self._global_plan_world_coord:
                        wp = CarlaDataProvider.get_map().get_waypoint(
                            transform.location,
                            project_to_road=True,
                            lane_type=carla.LaneType.Driving
                        )
                        plan.append((wp, cmd))
                    self._agent.set_global_plan(plan)
                    self._route_assigned = True

        # Get control from CARLA autopilot
        if self._agent:
            control = self._agent.run_step()
        else:
            # Fallback: no control
            control = carla.VehicleControl()
            control.steer = 0.0
            control.throttle = 0.0
            control.brake = 1.0

        # Save data using BaseAgent's functionality
        self.step += 1
        
        if hasattr(self, 'save_freq') and self.step % self.save_freq == 0:
            # Call parent's tick method to process sensor data
            if hasattr(self, 'tick'):
                self.tick(input_data)
            
            # Extract control values
            steer = control.steer
            throttle = control.throttle
            brake = 1.0 if control.brake > 0.5 else 0.0
            
            # Get route planning info
            gps = input_data["gps"][1][:2]
            pos = self._get_position({"gps": gps})
            
            # Get command from route planner
            if hasattr(self, '_command_planner') and self._command_planner and hasattr(self._command_planner, 'route') and self._command_planner.route:
                near_node, near_command = self._command_planner.run_step(pos)
                
                # Use waypoint planner if available, otherwise use command planner
                if hasattr(self, '_waypoint_planner') and self._waypoint_planner:
                    far_node, far_command = self._waypoint_planner.run_step(pos)
                else:
                    far_node, far_command = near_node, near_command
                
                target_speed = 20.0  # Default target speed
                
                # Save data
                if hasattr(self, 'save'):
                    self.save(
                        near_node,
                        far_node,
                        near_command,
                        steer,
                        throttle,
                        brake,
                        target_speed,
                        input_data,
                    )

        return control


