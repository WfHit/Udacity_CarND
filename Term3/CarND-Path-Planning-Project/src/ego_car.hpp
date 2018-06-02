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

#ifndef EGO_CAR_H_
#define EGO_CAR_H_

#include <math.h>
#include <vector>

#include "navigation.hpp"

class C_EgoCar_t
{
public:
  C_EgoCar_t();
  ~C_EgoCar_t(){;}
 	/**
   * @brief:update ego car state. 
   */  
  void UpdateCarInfo(double car_x, double car_y, double car_s, double car_d, double car_yaw, double car_speed);
 	/**
   * @brief:update ego car lane. 
   */  
  void SetLane(LANE_t lane) { m_current_lane = lane; }
 	/**
   * @brief:return ego car lane. 
   */  
  LANE_t GetCurrentLane() { return m_current_lane; }
 	/**
   * @brief:return ego car x. 
   */  
  double GetCartesianX() { return m_cartesian_x; }
 	/**
   * @brief:return ego car y. 
   */  
  double GetCartesianY() { return m_cartesian_y; }
 	/**
   * @brief:return ego car s. 
   */
  double GetFrenetS() { return m_frenet_s; }
 	/**
   * @brief:return ego car d. 
   */  
  double GetFrenetD() { return m_frenet_d; }
 	/**
   * @brief:return ego car line speed. 
   */
  double GetLineSpeed() { return m_line_speed; }
 	/**
   * @brief:return ego car yaw speed. 
   */  
  double GetYawSpeed() { return m_yaw_speed; }
 	/**
   * @brief:return sample period. 
   */  	
  double GetSamplePeriod() { return 0.02f; }
 	/**
   * @brief:return trajectory size. 
   */  
	int GetTrajectorySize() { return m_trajectory_size; }
	 /**
   * @brief:set trajectory size. 
   */  
	int SetTrajectorySize(int traj_size) { m_trajectory_size = traj_size; }
	 
  std::vector<double> m_start_s;
  std::vector<double> m_start_d;
	std::vector<double> m_end_s;
  std::vector<double> m_end_d;
  
private:
  LANE_t m_current_lane;
	double m_cartesian_x;
	double m_cartesian_y;
	double m_frenet_s;
	double m_frenet_d;
	double m_yaw_speed;
	double m_line_speed;
	int m_trajectory_size;
};

#endif /* EGO_CAR_H_ */

