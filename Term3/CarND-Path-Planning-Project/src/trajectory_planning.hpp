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
 
#ifndef TRAJECTORY_PLANNING_H_
#define TRAJECTORY_PLANNING_H_

#include <math.h>
#include <vector>
#include "Eigen-3.3/Eigen/Core"
#include "Eigen-3.3/Eigen/QR"
#include "Eigen-3.3/Eigen/Dense"

#include "ego_car.hpp"
#include "navigation.hpp"
#include "traffic_lane.hpp"

/**
* @class name: C_TrajectoryPlanning_t
* @brief: 
*/
class C_TrajectoryPlanning_t {
public:
  C_TrajectoryPlanning_t(C_Navigation_t* nav_ptr, C_EgoCar_t* egocar_ptr);
  ~C_TrajectoryPlanning_t(){;}
  /**
   * @brief: plan trajectory according to (start_s end_s) and (start_d end_d) with JMT. 
   * @note: result will add to the tail of (trajectory_s, trajectory_d)
   */ 
  void PlanTrajectory(std::vector<double>& trajectory_s, std::vector<double>& trajectory_d);

private:
  C_Navigation_t * m_navigation_ptr;
  C_EgoCar_t * m_egocar_ptr;
};
#endif /* TRAJECTORY_PLANNING_H_ */
