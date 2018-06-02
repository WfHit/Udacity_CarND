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
 
#ifndef TRAFFIC_LANE_H_
#define TRAFFIC_LANE_H_

#include "navigation.hpp"

typedef struct 
{
  int car_id;             //car's unique ID, 
  double cartesian_x;     //car's x position in map coordinates, 
  double cartesian_y;     //car's y position in map coordinates, 
  double velocity_x;      //car's x velocity in m/s, 
  double velocity_y;      //car's y velocity in m/s, 
  double velocity_total;  //car's total velocity in m/s, 
  double frenet_s;        //car's s position in frenet coordinates, 
  double frenet_d;        //car's d position in frenet coordinates. 
}CAR_t;

/**
* @class name: C_TrafficLane_t
* @brief: traffic lane contain a list discribles the other cars in this lane
*/

class C_TrafficLane_t 
{
public:
  C_TrafficLane_t(C_Navigation_t * nav_ptr);
  ~C_TrafficLane_t() {;}

  /**
   * @brief: add car to its lane.  
   */ 
  void AddCarInLane(CAR_t car) { m_cars_in_lane.push_back(car); }
  /**
   * @brief: clear cars in this lane.  
   */ 
  void ClearCarInLane() { m_cars_in_lane.clear(); }
  /**
   * @brief: set frenet D of current lane's middle line.  
   */   
  void SetCenterLineFrenetD(double frenet_d);
  /**
   * @brief: return the speed to the first front car 
   */
  double GetLaneFrontSpeed(double frenet_s);
  /**
   * @brief:return the distance to the first front car  
   */  
  double GetLaneFrontDistance(double frenet_s);
  /**
   * @brief:return the distance to the first back car  
   */  
  double GetLaneBackDistance(double frenet_s);
  /**
   * @brief: return the frenet D of the middle line.  
   */ 
  double GetCenterLineFrenetD();

private:
  double m_frenet_d;
  C_Navigation_t * m_navigation_ptr;
  std::vector<CAR_t> m_cars_in_lane;
  
};

#endif /* TRAFFIC_LANE_H_ */
