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
 
#include "behavioral_decision.hpp"

C_BehavioralDecision_t::C_BehavioralDecision_t(C_TrafficLane_t* left_lane_ptr, 
                                               C_TrafficLane_t* middle_lane_ptr, 
                                               C_TrafficLane_t* right_lane_ptr, 
                                               C_Navigation_t* navigation_ptr, 
                                               C_EgoCar_t* egocar_ptr) 
{
  m_left_lane_ptr = left_lane_ptr;
  m_middle_lane_ptr = middle_lane_ptr;
  m_right_lane_ptr = right_lane_ptr;
  m_navigation_ptr = navigation_ptr;
  m_egocar_ptr = egocar_ptr;
  m_b_cool_start = true;
	m_free_distance = 50.0f;
	m_safe_distance = 20.0f;
	m_lane_speed_weight = 50.0f;
	m_lane_obstacle_weight = 20.0f;
}

void C_BehavioralDecision_t::DecideBehavior()
{ 
  LANE_t current_lane = m_egocar_ptr->GetCurrentLane();
  if(m_b_cool_start) {
		//if first time run use policy of cool start
    m_b_cool_start = false;
    cool_start();
  } else {
    //check whether the current lane is blocked
    bool is_free = check_current_lane_free(m_egocar_ptr->GetCurrentLane());
    if(is_free) {
      //if not blocked keep in current lane use policy of free go
      free_go();
    } else {
      //if blocked try to find other lane
      LANE_t target_lane = find_target_lane();
      if(target_lane != current_lane) {
        //if target lane is not current lane, then change lane
        change_lane(target_lane);
      } else {
        //if target lane is  current lane, then following front car
        follow_car();
      }
    }
  }
}

C_TrafficLane_t * C_BehavioralDecision_t::get_lane_by_name(LANE_t lane_name) 
{
  C_TrafficLane_t * traffic_lane_ptr = NULL;
  if(LEFT_LANE == lane_name) {
    traffic_lane_ptr = m_left_lane_ptr;
  } else if(MIDDLE_LANE == lane_name) {
    traffic_lane_ptr = m_middle_lane_ptr;
  } else if(RIGHT_LANE == lane_name) {
    traffic_lane_ptr = m_right_lane_ptr;
  }
  return traffic_lane_ptr;
}

bool C_BehavioralDecision_t::check_current_lane_free(LANE_t lane)
{
	// Check if there is a vehicle block ego car
  bool is_free = true;
  C_TrafficLane_t * lane_ptr = get_lane_by_name(lane);
  if(lane_ptr->GetLaneFrontDistance(m_egocar_ptr->GetFrenetS()) < m_free_distance)
    is_free = false;
  return is_free;
}

bool C_BehavioralDecision_t::check_current_lane_safe(LANE_t lane)
{
	// Check if there is a vehicle block ego car
  bool is_free = true;
  C_TrafficLane_t * lane_ptr = get_lane_by_name(lane);
  if(lane_ptr->GetLaneFrontDistance(m_egocar_ptr->GetFrenetS()) < m_safe_distance)
    is_free = false;
  if(lane_ptr->GetLaneBackDistance(m_egocar_ptr->GetFrenetS()) < m_safe_distance)
    is_free = false;
  return is_free;
}

LANE_t C_BehavioralDecision_t::find_target_lane()
{
  LANE_t low_cost_lane = find_low_cost_lane();
  if(check_lane_feasible(low_cost_lane))
    return low_cost_lane;
  else 
    return m_egocar_ptr->GetCurrentLane();
}

LANE_t C_BehavioralDecision_t::find_low_cost_lane()
{
  double left_lane_cost = cal_lane_cost(LEFT_LANE);
  double middle_lane_cost = cal_lane_cost(MIDDLE_LANE);
  double right_lane_cost = cal_lane_cost(RIGHT_LANE);
  
  double lowest_cost = left_lane_cost;
  LANE_t target_lane = LEFT_LANE;

  if (middle_lane_cost < lowest_cost)
  {
    lowest_cost = middle_lane_cost;
    target_lane = MIDDLE_LANE;
  }
  if (right_lane_cost < lowest_cost)
  {
    lowest_cost = right_lane_cost;
    target_lane = RIGHT_LANE;
  }
	
  return target_lane;
}

bool C_BehavioralDecision_t::check_lane_feasible(LANE_t target_lane)
{
	bool b_safe = true;
	LANE_t current_lane = m_egocar_ptr->GetCurrentLane();
	//target lane is feasible when target lane is free 
	if(!check_current_lane_safe(target_lane))
		b_safe = false;
	//target lane is feasible when target lane next to current lane 
	if(fabs((int)current_lane - (int)target_lane) > 1)
		b_safe = false;
	return b_safe;
}
  
// Adjust the speed in free go
double C_BehavioralDecision_t::target_speed_free_go(double front_distance, double front_speed)
{
  double speed_factor = 1.00f;

	if(front_distance > m_free_distance)
		speed_factor = 1.05f;
	if(front_distance > 2 * m_free_distance)
		speed_factor = 1.10f;
  if(front_distance > 4 * m_free_distance)
		speed_factor = 1.2f;	
		
	double target_speed = m_egocar_ptr->m_end_s[1] * speed_factor;
	
  return std::min(target_speed,  m_navigation_ptr->GetReferenceSpeed());
}

// Adjust the speed in while follow car
double C_BehavioralDecision_t::target_speed_follow_car( double front_distance, double front_speed)
{
	double speed_factor = 1.00f;
  double ego_car_current_speed = m_egocar_ptr->GetLineSpeed();
  if (front_distance > m_safe_distance)
  {
    speed_factor = 0.98f;
  }
  if (front_distance > (m_safe_distance+0.5*(m_free_distance-m_safe_distance)))
  { 
    if(front_speed > ego_car_current_speed)
      speed_factor = 1.05f;
    if(front_speed < ego_car_current_speed)
      speed_factor = 1.00f;
  }
  if (front_distance < 0.80 * m_safe_distance)
  {
    speed_factor = 0.95;
  }
	
	double target_speed = m_egocar_ptr->m_end_s[1] * speed_factor;

  return std::min(target_speed,  m_navigation_ptr->GetReferenceSpeed());
}

double C_BehavioralDecision_t::target_speed_change_lane( double front_distance, double front_speed)
{
  double target_speed = m_egocar_ptr->m_end_s[1];

  if (front_distance < m_free_distance)
  {
    target_speed = 0.9 * target_speed;
  }
  
  target_speed = std::min(target_speed, front_speed);
  
  return std::min(target_speed,  m_navigation_ptr->GetReferenceSpeed());
}

double C_BehavioralDecision_t::get_lane_front_speed(LANE_t lane) 
{	
  C_TrafficLane_t * lane_ptr = get_lane_by_name(lane);
  return lane_ptr->GetLaneFrontSpeed(m_egocar_ptr->GetFrenetS());
}

double C_BehavioralDecision_t::get_lane_front_distance(LANE_t lane)
{
  C_TrafficLane_t * lane_ptr = get_lane_by_name(lane);
  return lane_ptr->GetLaneFrontDistance(m_egocar_ptr->GetFrenetS());
}

double C_BehavioralDecision_t::get_lane_center_d(LANE_t lane)
{
  C_TrafficLane_t * lane_ptr = get_lane_by_name(lane);
  return lane_ptr->GetCenterLineFrenetD();
}

// Start state action
void C_BehavioralDecision_t::cool_start()
{
	m_egocar_ptr->SetTrajectorySize(150);
  double sample_period = m_egocar_ptr->GetSamplePeriod();
  double front_v = get_lane_front_speed(m_egocar_ptr->GetCurrentLane());
  double target_v = std::min(front_v, m_navigation_ptr->GetReferenceSpeed() / 4.0);
  double target_s = m_egocar_ptr->GetFrenetS() + m_egocar_ptr->GetTrajectorySize() * sample_period * target_v;

  m_egocar_ptr->m_start_s = {m_egocar_ptr->GetFrenetS(), 0.0, 0.0};
  m_egocar_ptr->m_end_s= {target_s, target_v, 0.0};
  m_egocar_ptr->m_start_d = {get_lane_center_d(m_egocar_ptr->GetCurrentLane()), 0.0, 0.0};
  m_egocar_ptr->m_end_d = {get_lane_center_d(m_egocar_ptr->GetCurrentLane()), 0.0, 0.0};

	std::cout << "*************************************************" << std::endl;
	std::cout << "POLICY: Cool Start " << std::endl;
	std::cout << "-------------------------------------------------" << std::endl;
	std::cout << "S START: [" << m_egocar_ptr->m_start_s[0] << " -- " << m_egocar_ptr->m_start_s[1] << " -- " << m_egocar_ptr->m_start_s[2] << "]" << std::endl;
	std::cout << "S END  : [" << m_egocar_ptr->m_end_s[0] << " -- " << m_egocar_ptr->m_end_s[1] << " -- " << m_egocar_ptr->m_end_s[2] << "]" << std::endl;
	std::cout << "D START: [" << m_egocar_ptr->m_start_d[0] << " -- " << m_egocar_ptr->m_start_d[1] << " -- " << m_egocar_ptr->m_start_d[2] << "]" << std::endl;
	std::cout << "D END  : [" << m_egocar_ptr->m_end_d[0] << " -- " << m_egocar_ptr->m_end_d[1] << " -- " << m_egocar_ptr->m_end_d[2] << "]" << std::endl;
	std::cout << "*************************************************" << std::endl;

}

// Keep lane action
void C_BehavioralDecision_t::free_go()
{
	m_egocar_ptr->SetTrajectorySize(100);
  double sample_period = m_egocar_ptr->GetSamplePeriod();
  double front_v = get_lane_front_speed(m_egocar_ptr->GetCurrentLane());
  double front_distance = get_lane_front_distance(m_egocar_ptr->GetCurrentLane());
  double target_v = target_speed_free_go(front_distance, front_v);
  double target_s = m_egocar_ptr->m_end_s[0] + m_egocar_ptr->GetTrajectorySize() * sample_period * target_v;
  double target_d = get_lane_center_d(m_egocar_ptr->GetCurrentLane());

  m_egocar_ptr->m_start_s = {m_egocar_ptr->m_end_s[0], m_egocar_ptr->m_end_s[1], m_egocar_ptr->m_end_s[2]};
  m_egocar_ptr->m_end_s = {target_s, target_v, 0.0};
  m_egocar_ptr->m_start_d = {m_egocar_ptr->m_end_d[0], 0.0, 0.0};
  m_egocar_ptr->m_end_d = {target_d, 0.0, 0.0};

	std::cout << "*************************************************" << std::endl;
	std::cout << "POLICY: Free Go " << std::endl;
	std::cout << "-------------------------------------------------" << std::endl;
	std::cout << "S START: [" << m_egocar_ptr->m_start_s[0] << " -- " << m_egocar_ptr->m_start_s[1] << " -- " << m_egocar_ptr->m_start_s[2] << "]" << std::endl;
	std::cout << "S END  : [" << m_egocar_ptr->m_end_s[0] << " -- " << m_egocar_ptr->m_end_s[1] << " -- " << m_egocar_ptr->m_end_s[2] << "]" << std::endl;
	std::cout << "D START: [" << m_egocar_ptr->m_start_d[0] << " -- " << m_egocar_ptr->m_start_d[1] << " -- " << m_egocar_ptr->m_start_d[2] << "]" << std::endl;
	std::cout << "D END  : [" << m_egocar_ptr->m_end_d[0] << " -- " << m_egocar_ptr->m_end_d[1] << " -- " << m_egocar_ptr->m_end_d[2] << "]" << std::endl;
	std::cout << "*************************************************" << std::endl;

}

// Slow down action
void C_BehavioralDecision_t::follow_car()
{
	m_egocar_ptr->SetTrajectorySize(80);
  double sample_period = m_egocar_ptr->GetSamplePeriod();
  double front_v = get_lane_front_speed(m_egocar_ptr->GetCurrentLane());
  double front_distance = get_lane_front_distance(m_egocar_ptr->GetCurrentLane());
  double target_v = target_speed_follow_car(front_distance, front_v);
  double target_s = m_egocar_ptr->m_end_s[0] + m_egocar_ptr->GetTrajectorySize() * sample_period * target_v;
  double target_d = get_lane_center_d(m_egocar_ptr->GetCurrentLane());

  m_egocar_ptr->m_start_s = {m_egocar_ptr->m_end_s[0], m_egocar_ptr->m_end_s[1], m_egocar_ptr->m_end_s[2]};
  m_egocar_ptr->m_end_s = {target_s, target_v, 0.0};
  m_egocar_ptr->m_start_d = {m_egocar_ptr->m_end_d[0], 0.0, 0.0};
  m_egocar_ptr->m_end_d = {target_d, 0.0, 0.0};
	
	std::cout << "*************************************************" << std::endl;
	std::cout << "POLICY: Follow Car " << std::endl;
	std::cout << "-------------------------------------------------" << std::endl;
	std::cout << "S START: [" << m_egocar_ptr->m_start_s[0] << " -- " << m_egocar_ptr->m_start_s[1] << " -- " << m_egocar_ptr->m_start_s[2] << "]" << std::endl;
	std::cout << "S END  : [" << m_egocar_ptr->m_end_s[0] << " -- " << m_egocar_ptr->m_end_s[1] << " -- " << m_egocar_ptr->m_end_s[2] << "]" << std::endl;
	std::cout << "D START: [" << m_egocar_ptr->m_start_d[0] << " -- " << m_egocar_ptr->m_start_d[1] << " -- " << m_egocar_ptr->m_start_d[2] << "]" << std::endl;
	std::cout << "D END  : [" << m_egocar_ptr->m_end_d[0] << " -- " << m_egocar_ptr->m_end_d[1] << " -- " << m_egocar_ptr->m_end_d[2] << "]" << std::endl;
	std::cout << "*************************************************" << std::endl;
	
}

// Change lane action
void C_BehavioralDecision_t::change_lane(LANE_t target_lane)
{
	m_egocar_ptr->SetTrajectorySize(150);
  double sample_period = m_egocar_ptr->GetSamplePeriod();
  double front_v = get_lane_front_speed(target_lane);
  double front_distance = get_lane_front_distance(target_lane);
  double target_v = target_speed_change_lane(front_distance, front_distance);
  double target_s = m_egocar_ptr->m_end_s[0] + m_egocar_ptr->GetTrajectorySize() * sample_period * target_v;
  double target_d = get_lane_center_d(target_lane);

  m_egocar_ptr->m_start_s = {m_egocar_ptr->m_end_s[0], m_egocar_ptr->m_end_s[1], m_egocar_ptr->m_end_s[2]};
  m_egocar_ptr->m_end_s = {target_s, target_v, 0.0};
  m_egocar_ptr->m_start_d = {m_egocar_ptr->m_end_d[0], 0.0, 0.0};
  m_egocar_ptr->m_end_d = {target_d, 0.0, 0.0};
	
	std::cout << "*************************************************" << std::endl;
	std::cout << "POLICY: Change Lane " << std::endl;
	std::cout << "-------------------------------------------------" << std::endl;
	std::cout << "S START: [" << m_egocar_ptr->m_start_s[0] << " -- " << m_egocar_ptr->m_start_s[1] << " -- " << m_egocar_ptr->m_start_s[2] << "]" << std::endl;
	std::cout << "S END  : [" << m_egocar_ptr->m_end_s[0] << " -- " << m_egocar_ptr->m_end_s[1] << " -- " << m_egocar_ptr->m_end_s[2] << "]" << std::endl;
	std::cout << "D START: [" << m_egocar_ptr->m_start_d[0] << " -- " << m_egocar_ptr->m_start_d[1] << " -- " << m_egocar_ptr->m_start_d[2] << "]" << std::endl;
	std::cout << "D END  : [" << m_egocar_ptr->m_end_d[0] << " -- " << m_egocar_ptr->m_end_d[1] << " -- " << m_egocar_ptr->m_end_d[2] << "]" << std::endl;
	std::cout << "*************************************************" << std::endl;
	
}

double C_BehavioralDecision_t::lane_speed_cost(LANE_t lane)
{
  C_TrafficLane_t * lane_ptr = get_lane_by_name(lane);
  double lane_speed = lane_ptr->GetLaneFrontSpeed(m_egocar_ptr->GetFrenetS());
  return 1.0 / lane_speed;
}


// Find the lane whose closer vehicles is further
double C_BehavioralDecision_t::lane_distance_cost(LANE_t lane)
{
  C_TrafficLane_t * lane_ptr = get_lane_by_name(lane);
  double distance_front = lane_ptr->GetLaneFrontDistance(m_egocar_ptr->GetFrenetS());
	double distance_back = lane_ptr->GetLaneBackDistance(m_egocar_ptr->GetFrenetS());
	double distance = distance_front<distance_back ? distance_front : distance_back;
  return 1.0 / distance;
}

double C_BehavioralDecision_t::cal_lane_cost(LANE_t lane)
{
  double speed_cost = m_lane_speed_weight * lane_speed_cost(lane);
  double distance_cost = m_lane_obstacle_weight * lane_distance_cost(lane);
  return speed_cost + distance_cost;
}
