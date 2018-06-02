/******************************************************************************
 * Copyright 2018 Frank Wong. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *****************************************************************************/
 
#ifndef BEHAVIORAL_DECISION_H_
#define BEHAVIORAL_DECISION_H_

#include "ego_car.hpp"
#include "traffic_lane.hpp"
/**
* @class name: C_BehavioralDecision_t
* @brief: decide which lane should the car follow
*/

class C_BehavioralDecision_t 
{
public:
  C_BehavioralDecision_t(C_TrafficLane_t* left_lane_ptr, 
                         C_TrafficLane_t* middle_lane_ptr, 
                         C_TrafficLane_t* right_lane_ptr, 
                         C_Navigation_t* navigation_ptr, 
                         C_EgoCar_t* egocar_ptr);
  ~C_BehavioralDecision_t() {;}
 	/**
   * @brief:decide behavior
   * note:here I'm not use state machine. In the fact, 
   * I use a design pattern called strategy pattern. 
   * First, the ego car check the current lane to 
   * make sure whether it is blocked by other cars, 
   * if no cars in front of ego car, then ego car adopt 
   * free go policy;
   * If ego car is blocked by other cars, it will try to 
   * find the lowest cost line. If the lowest cost line 
   * is other line, then it will adopt change line policy,
   * if the lowest cost line is current line, it will adopt 
   * follow car policy. 
   */    
  void DecideBehavior();
  
private:
 	/**
   * @brief:return lane specified by name
   */ 
	C_TrafficLane_t * get_lane_by_name(LANE_t lane_name);
	LANE_t find_target_lane();
	LANE_t find_low_cost_lane();
	
	bool check_current_lane_free(LANE_t lane);
	bool check_lane_feasible(LANE_t lane);
	bool check_current_lane_safe(LANE_t lane);
  void cool_start();
  void free_go();
  void follow_car();
  void change_lane(LANE_t target_lane);
  double target_speed_free_go(double distance, double speed);
  double target_speed_follow_car( double distance, double speed);
  double target_speed_change_lane(double distance, double speed);
	double get_lane_front_speed(LANE_t lane);
  double get_lane_front_distance(LANE_t lane);
  double get_lane_center_d(LANE_t lane);
	double cal_lane_cost(LANE_t lane);
  double lane_speed_cost(LANE_t lane);
  double lane_distance_cost(LANE_t lane);
	
  C_TrafficLane_t * m_left_lane_ptr;
  C_TrafficLane_t * m_middle_lane_ptr;
  C_TrafficLane_t * m_right_lane_ptr;
  C_Navigation_t * m_navigation_ptr;
  C_EgoCar_t * m_egocar_ptr;
  
  bool m_b_cool_start;
	double m_free_distance;	//beyond m_free_distance we thought road if free
	double m_safe_distance;	//beyond m_safe_distance we thought road if safe
	double m_lane_speed_weight;
	double m_lane_obstacle_weight;
};

#endif /* BEHAVIORAL_DECISION_H_ */
