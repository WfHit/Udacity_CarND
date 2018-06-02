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
#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <sstream>
#include <math.h>

#include "navigation.hpp"

void C_Navigation_t::Initialize() 
{
  // Waypoint map to read from
  std::string map_file_ = "../data/highway_map.csv";

  std::ifstream in_map_(map_file_.c_str(), std::ifstream::in);

  std::string line;
	double x;
	double y;
	float s;
	float d_x;
	float d_y;
  while (getline(in_map_, line)) {
  	std::istringstream iss(line);
  	iss >> x;
  	iss >> y;
  	iss >> s;
  	iss >> d_x;
  	iss >> d_y;
  	m_map_waypoints_x.push_back(x);
  	m_map_waypoints_y.push_back(y);
  	m_map_waypoints_s.push_back(s);
  	m_map_waypoints_dx.push_back(d_x);
  	m_map_waypoints_dy.push_back(d_y);
  }
	
	in_map_.close();
  
  // add addtion data to make it closed
	x = 784.6001;
	y = 1135.571;
	s = 6945.554;
	d_x = -0.02359831;
	d_y = -0.9997216;

	m_map_waypoints_x.push_back(x);
	m_map_waypoints_y.push_back(y);
	m_map_waypoints_s.push_back(s);
	m_map_waypoints_dx.push_back(d_x);
	m_map_waypoints_dy.push_back(d_y);

  m_lane_spline_x.set_points(m_map_waypoints_s, m_map_waypoints_x); 
  m_lane_spline_y.set_points(m_map_waypoints_s, m_map_waypoints_y); 
  m_lane_spline_dx.set_points(m_map_waypoints_s, m_map_waypoints_dx); 
  m_lane_spline_dy.set_points(m_map_waypoints_s, m_map_waypoints_dy); 
}

std::vector<double> C_Navigation_t::GetSplineXY(double s, double d)
{
  double x, y, dx, dy;
  x = m_lane_spline_x(s);
  y = m_lane_spline_y(s);
  dx = m_lane_spline_dx(s);
  dy = m_lane_spline_dy(s); 
   
  double cart_x, cart_y;
  cart_x = x + dx * d;
  cart_y = y + dy * d;
  
	return {cart_x,cart_y};
}
