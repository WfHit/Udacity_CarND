#include "PID.h"

using namespace std;

/*
* TODO: Complete the PID class.
*/

PID::PID() {}

PID::~PID() {}

void PID::Init(double Kp, double Ki, double Kd) {
  // initial paramters
  this->Kp = Kp;
  this->Ki = Ki;
  this->Kd = Kd;
  // initial states
  p_error = 0.0;
  i_error = 0.0;
  d_error = 0.0;
  last_cte = 0.0;
}

void PID::UpdateError(double cte) {
  //update error
  p_error = cte;
  d_error = cte - last_cte;
  i_error += cte;
  if(i_error > 10000000.0) i_error = 10000000.0;
  if(i_error < -10000000.0) i_error = -10000000.0; 
  last_cte = cte;
  
}

double PID::TotalError() {
  // calculate PID output
  double control_output = Kp * p_error + Ki * i_error + Kd * d_error;
  
  return control_output;
}

