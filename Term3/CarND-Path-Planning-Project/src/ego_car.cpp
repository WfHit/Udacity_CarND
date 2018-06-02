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

#include "ego_car.hpp"

C_EgoCar_t::C_EgoCar_t() {
	m_current_lane = ANY_LANE;
	m_cartesian_x = 0.0;
	m_cartesian_y = 0.0;
	m_frenet_s = 0.0;
	m_frenet_d = 0.0;
	m_yaw_speed = 0.0;
	m_line_speed = 0.0;
	m_trajectory_size = 120;
}

void C_EgoCar_t::UpdateCarInfo(double car_x, double car_y, double car_s, double car_d, double car_yaw, double car_speed) {
  m_cartesian_x = car_x; 
  m_cartesian_y = car_y; 
  m_frenet_s = car_s; 
  m_frenet_d = car_d; 
  m_yaw_speed = car_yaw; 
  m_line_speed = car_speed;
}

