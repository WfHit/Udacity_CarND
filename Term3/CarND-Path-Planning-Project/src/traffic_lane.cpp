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
 
#include "traffic_lane.hpp" 

C_TrafficLane_t::C_TrafficLane_t(C_Navigation_t * nav_ptr) 
{
  m_navigation_ptr = nav_ptr;
}

double C_TrafficLane_t::GetLaneFrontSpeed(double frenet_s) 
{
  double front_car_speed = 1e5;
  double front_car_distance = 1e5;
	
  for (int counter=0; counter<m_cars_in_lane.size(); counter++) {
    double distance = m_cars_in_lane[counter].frenet_s - frenet_s;
    if(distance > 0 && distance < front_car_distance) {
      front_car_distance = distance;
			front_car_speed = m_cars_in_lane[counter].velocity_total;
    }
  }
  return front_car_speed;
}

double C_TrafficLane_t::GetLaneFrontDistance(double frenet_s)
{
  double front_car_distance = 1e5;
	
  for (int counter=0; counter<m_cars_in_lane.size(); counter++) {
    double distance = m_cars_in_lane[counter].frenet_s - frenet_s;
    if(distance > 0 && distance < front_car_distance) {
      front_car_distance = distance;
    }
  }
  return front_car_distance;
}

double C_TrafficLane_t::GetLaneBackDistance(double frenet_s)
{
  double back_car_distance = 1e5;
	
  for (int counter=0; counter<m_cars_in_lane.size(); counter++) {
    double distance = -(m_cars_in_lane[counter].frenet_s - frenet_s);
    if(distance > 0 && distance < back_car_distance) {
      back_car_distance = distance;
    }
  }
  return back_car_distance;
}


double C_TrafficLane_t::GetCenterLineFrenetD() 
{
  return m_frenet_d;
}

void C_TrafficLane_t::SetCenterLineFrenetD(double frenet_d)
{
  m_frenet_d = frenet_d;
}


