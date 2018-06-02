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

#include "trajectory_planning.hpp"

using Eigen::MatrixXd;
using Eigen::VectorXd;

std::vector<double> JMT(std::vector< double> start, std::vector <double> end, double T)
{
  /*
  Calculate the Jerk Minimizing Trajectory that connects the initial state
  to the final state in time T.

  INPUTS

  start - the vehicles start location given as a length three array
      corresponding to initial values of [s, s_dot, s_double_dot]

  end   - the desired end state for vehicle. Like "start" this is a
      length three array.

  T     - The duration, in seconds, over which this maneuver should occur.

  OUTPUT 
  an array of length 6, each value corresponding to a coefficent in the polynomial 
  s(t) = a_0 + a_1 * t + a_2 * t**2 + a_3 * t**3 + a_4 * t**4 + a_5 * t**5

  EXAMPLE

  > JMT( [0, 10, 0], [10, 10, 0], 1)
  [0.0, 10.0, 0.0, 0.0, 0.0, 0.0]
  */
  
  MatrixXd A = MatrixXd(3, 3);
	A << T*T*T, T*T*T*T, T*T*T*T*T,
			    3*T*T, 4*T*T*T,5*T*T*T*T,
			    6*T, 12*T*T, 20*T*T*T;
		
	MatrixXd B = MatrixXd(3,1);	    
	B << end[0]-(start[0]+start[1]*T+.5*start[2]*T*T),
			    end[1]-(start[1]+start[2]*T),
			    end[2]-start[2];
			    
	MatrixXd Ai = A.inverse();
	
	MatrixXd C = Ai*B;
	
	std::vector <double> result = {start[0], start[1], .5*start[2]};
	for(int i = 0; i < C.size(); i++)
	{
	    result.push_back(C.data()[i]);
	}
	
  return result;
    
}

C_TrajectoryPlanning_t::C_TrajectoryPlanning_t(C_Navigation_t* nav_ptr, C_EgoCar_t* egocar_ptr) 
{
  m_navigation_ptr = nav_ptr;
  m_egocar_ptr = egocar_ptr;
}

void C_TrajectoryPlanning_t::PlanTrajectory(std::vector<double>& trajectory_s, std::vector<double>& trajectory_d) 
{
	double sample_period = m_egocar_ptr->GetSamplePeriod();
	int tragectory_size = m_egocar_ptr->GetTrajectorySize();
  double jmt_time = tragectory_size * sample_period;
   
  // Compute Jerk Minimizing Trajectory s(t) and d(t)
  std::vector<double> s_jmt = JMT(m_egocar_ptr->m_start_s, m_egocar_ptr->m_end_s, jmt_time);
  std::vector<double> d_jmt = JMT(m_egocar_ptr->m_start_d, m_egocar_ptr->m_end_d, jmt_time);

  for(int counter=0; counter<tragectory_size; counter++) {
		
    double tick = sample_period * counter;

		// Evaluate s(t) at time sample_period * counter
		// s(t) = a_0 + a_1 * t + a_2 * t**2 + a_3 * t**3 + a_4 * t**4 + a_5 * t**5   
    double frenet_s = s_jmt[0] + 
											s_jmt[1] * pow(tick, 1) +
											s_jmt[2] * pow(tick, 2) +
											s_jmt[3] * pow(tick, 3) +
											s_jmt[4] * pow(tick, 4) +
											s_jmt[5] * pow(tick, 5);

    // Evaluate d(t) at time sample_period * counter
    // d(t) = a_0 + a_1 * t + a_2 * t**2 + a_3 * t**3 + a_4 * t**4 + a_5 * t**5  
    double frenet_d = d_jmt[0] + 
											d_jmt[1] * pow(tick, 1) +
											d_jmt[2] * pow(tick, 2) +
											d_jmt[3] * pow(tick, 3) +
											d_jmt[4] * pow(tick, 4) +
											d_jmt[5] * pow(tick, 5);

    // Wrap around s and d coordinates
    double frenet_s_mod = fmod(frenet_s, m_navigation_ptr->GetTotalLength());
    double frenet_d_mod = fmod(frenet_d, m_navigation_ptr->GetTotalWidth());

    // Convert Frenet coordinates to XY coordinates
    std::vector<double> x_y = m_navigation_ptr->GetSplineXY(frenet_s_mod, frenet_d_mod);
		
    trajectory_s.push_back(x_y[0]);
    trajectory_d.push_back(x_y[1]);
  }
}

