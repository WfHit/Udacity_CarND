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
 
#ifndef NAVIGATION_H_
#define NAVIGATION_H_

#include <fstream>
#include <math.h>
#include <chrono>
#include <iostream>
#include <thread>
#include <vector>
#include "spline.hpp"

typedef enum {
  ANY_LANE = 0,
  LEFT_LANE = 1,
  MIDDLE_LANE = 2,
  RIGHT_LANE = 3,  
  TOTAL_LANES = 4
}LANE_t;

/**
* @class name: C_Navigation_t
* @brief: 
*/

class C_Navigation_t 
{
public:
  C_Navigation_t(){;}
  ~C_Navigation_t(){;}
  /**
   * @brief:create spline of x, y, dx, dy ralate to s. 
   */  
  void Initialize();
	/**
   * @brief:calculate x, y with spline. 
   */  
  std::vector<double> GetSplineXY(double s, double d);
	/**
   * @brief:return maxmium feasible speed. 
   */  
  double GetReferenceSpeed() { return 21.0f; }
	/**
   * @brief:return cyclic length of road. 
   */ 
  double GetTotalLength() { return 6945.554f; }  // The max s value before wrapping around the track back to 0
	/**
   * @brief:return width of road. 
   */ 
  double GetTotalWidth() { return 12.0f; }
	
  std::vector<double> m_map_waypoints_x;
  std::vector<double> m_map_waypoints_y;
  std::vector<double> m_map_waypoints_s;
  std::vector<double> m_map_waypoints_dx;
  std::vector<double> m_map_waypoints_dy;
  
private:
  tk::spline m_lane_spline_x; 
  tk::spline m_lane_spline_y;
  tk::spline m_lane_spline_dx;
  tk::spline m_lane_spline_dy;
};
#endif /* NAVIGATION_H_ */
