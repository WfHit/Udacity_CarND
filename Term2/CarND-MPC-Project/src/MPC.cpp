#include "MPC.h"
#include <cppad/cppad.hpp>
#include <cppad/ipopt/solve.hpp>
#include "Eigen-3.3/Eigen/Core"

using CppAD::AD;

// TODO: Set the timestep length and duration
size_t N = 25;
double dt = 0.05;

// This value assumes the model presented in the classroom is used.
//
// It was obtained by measuring the radius formed by running the vehicle in the
// simulator around in a circle with a constant steering angle and velocity on a
// flat terrain.
//
// Lf was tuned until the the radius formed by the simulating the model
// presented in the classroom matched the previous radius.
//
// This is the length from front to CoG that has a similar radius.
const double Lf = 2.67;

// The solver takes all the state variables and actuator
// variables in a singular vector. Thus, we should to establish
// when one variable starts and another ends to make our lifes easier.
size_t x_start = 0;
size_t y_start = x_start + N;
size_t psi_start = y_start + N;
size_t v_start = psi_start + N;
size_t cte_start = v_start + N;
size_t epsi_start = cte_start + N;
size_t delta_start = epsi_start + N;
size_t a_start = delta_start + N - 1;

// reference value for cte, epsi, velocity
double ref_cte = 0;
double ref_epsi = 0;
double ref_speed = 60;

//Weights for cost
int weight_cte = 60;
int weight_delta = 100000;
int weight_delta_gap = 1000;
int weight_eps = 10;
int weight_speed = 10;
int weight_a = 20;
int weight_a_gap = 100;

class FG_eval {
 public:
  // Fitted polynomial coefficients
  Eigen::VectorXd coeffs;
  FG_eval(Eigen::VectorXd coeffs) { this->coeffs = coeffs; }

  typedef CPPAD_TESTVECTOR(AD<double>) ADvector;
  void operator()(ADvector& fg, const ADvector& vars) {
    // TODO: implement MPC
    // `fg` a vector of the cost constraints, `vars` is a vector of variable values (state & actuators)
    // NOTE: You'll probably go back and forth between this function and
    // the Solver function below.
    
    fg[0] = 0;

    // The part of the cost based on the reference state.
    for (size_t idx = 0; idx < N; idx++) {
      fg[0] += weight_cte * CppAD::pow(vars[cte_start + idx] - ref_cte, 2);
      fg[0] += weight_eps * CppAD::pow(vars[epsi_start + idx]- ref_epsi, 2);
      fg[0] += weight_speed * CppAD::pow(vars[v_start + idx] - ref_speed, 2);
    }

    // Minimize the use of actuators.
    for (size_t idx = 0; idx < N - 1; idx++) {
      fg[0] += weight_delta * CppAD::pow(vars[delta_start + idx], 2);
      fg[0] += weight_a * CppAD::pow(vars[a_start + idx], 2);
    }

    // Minimize the value gap between sequential actuations.
    for (size_t idx = 0; idx < N - 2; idx++) {
      fg[0] += weight_delta_gap * CppAD::pow(vars[delta_start + idx + 1] - vars[delta_start + idx], 2);
      fg[0] += weight_a_gap * CppAD::pow(vars[a_start + idx + 1] - vars[a_start + idx], 2);
    }

    // Setup Constraints
    // We add 1 to each of the starting indices due to cost being located at index 0 of `fg`.
    // This bumps up the position of all the other values.
    fg[1 + x_start] = vars[x_start];
    fg[1 + y_start] = vars[y_start];
    fg[1 + psi_start] = vars[psi_start];
    fg[1 + v_start] = vars[v_start];
    fg[1 + cte_start] = vars[cte_start];
    fg[1 + epsi_start] = vars[epsi_start];

    // The rest of the constraints
    for (size_t idx = 1; idx < N; idx++) {
      // The state at time idx+1 .
      AD<double> x1 = vars[x_start + idx];
      AD<double> y1 = vars[y_start + idx];
      AD<double> psi1 = vars[psi_start + idx];
      AD<double> v1 = vars[v_start + idx];
      AD<double> cte1 = vars[cte_start + idx];
      AD<double> epsi1 = vars[epsi_start + idx];

      // The state at time idx.
      AD<double> x0 = vars[x_start + idx - 1];
      AD<double> y0 = vars[y_start + idx - 1];
      AD<double> psi0 = vars[psi_start + idx - 1];
      AD<double> v0 = vars[v_start + idx - 1];
      AD<double> cte0 = vars[cte_start + idx - 1];
      AD<double> epsi0 = vars[epsi_start + idx - 1];

      // Only consider the actuation at time idx.
      AD<double> delta0 = vars[delta_start + idx - 1];
      AD<double> a0 = vars[a_start + idx - 1];

      AD<double> f0 = coeffs[0] + coeffs[1] * x0 + coeffs[2] * x0 * x0 + coeffs[3] * x0 * x0 *x0;
      AD<double> psides0 = CppAD::atan(3 * coeffs[3] * x0 * x0 + 2 * coeffs[2] * x0 + coeffs[1]);

      // Here's `x` to get you started.
      // The idea here is to constraint this value to be 0.
      //
      // Recall the equations for the model:
      // x_[idx+1] = x[idx] + v[idx] * cos(psi[idx]) * dt
      // y_[idx+1] = y[idx] + v[idx] * sin(psi[idx]) * dt
      // psi_[idx+1] = psi[idx] + v[idx] / Lf * delta[idx] * dt
      // v_[idx+1] = v[idx] + a[idx] * dt
      // cte[idx+1] = f(x[idx]) - y[idx] + v[idx] * sin(epsi[idx]) * dt
      // epsi[idx+1] = psi[idx] - psides[idx] + v[idx] * delta[idx] / Lf * dt
      fg[1 + x_start + idx] = x1 - (x0 + v0 * CppAD::cos(psi0) * dt);
      fg[1 + y_start + idx] = y1 - (y0 + v0 * CppAD::sin(psi0) * dt);
      fg[1 + psi_start + idx] = psi1 - (psi0 + v0 * delta0 / Lf * dt);
      fg[1 + v_start + idx] = v1 - (v0 + a0 * dt);
      fg[1 + cte_start + idx] = cte1 - ((f0 - y0) + (v0 * CppAD::sin(epsi0) * dt));
      fg[1 + epsi_start + idx] = epsi1 - ((psi0 - psides0) + v0 * delta0 / Lf * dt);
    }
  }
};

//
// MPC class definition implementation.
//
MPC::MPC() {}
MPC::~MPC() {}

vector<double> MPC::Solve(Eigen::VectorXd state, Eigen::VectorXd coeffs) {
  bool ok = true;

  typedef CPPAD_TESTVECTOR(double) ADvector;

  double x = state[0];
  double y = state[1];
  double psi = state[2];
  double v = state[3];
  double cte = state[4];
  double epsi = state[5];
  
  // TODO: Set the number of model variables (includes both states and inputs).
  // For example: If the state is a 4 element vector, the actuators is a 2
  // element vector and there are 10 timesteps. The number of variables is:
  //
  // number of independent variables
  // N timesteps == N - 1 actuations
  size_t n_vars = N * 6 + (N - 1) * 2;
  // Number of constraints
  size_t n_constraints = N * 6;

  // Initial value of the independent variables.
  // SHOULD BE 0 besides initial state.
  ADvector vars(n_vars);
  for (size_t idx = 0; idx < n_vars; idx++) {
    vars[idx] = 0;
  }

  // Set the initial variable values
  vars[x_start] = x;
  vars[y_start] = y;
  vars[psi_start] = psi;
  vars[v_start] = v;
  vars[cte_start] = cte;
  vars[epsi_start] = epsi;
  
  ADvector vars_lowerbound(n_vars);
  ADvector vars_upperbound(n_vars);
  // TODO: Set lower and upper limits for variables.
  // Set all non-actuators upper and lowerlimits
  // to the max negative and positive values.
  for (size_t idx = 0; idx < delta_start; idx++) {
    vars_lowerbound[idx] = -1.0e19;
    vars_upperbound[idx] = 1.0e19;
  }

  // The upper and lower limits of delta are set to -25 and 25
  // degrees (values in radians).
  // NOTE: Feel free to change this to something else.
  for (size_t idx = delta_start; idx < a_start; idx++) {
    vars_lowerbound[idx] = -0.436332;
    vars_upperbound[idx] = 0.436332;
  }

  // Acceleration/decceleration upper and lower limits.
  // NOTE: Feel free to change this to something else.
  for (size_t idx = a_start; idx < n_vars; idx++) {
    vars_lowerbound[idx] = -1.0;
    vars_upperbound[idx] = 1.0;
  }

  // Lower and upper limits for the constraints
  // Should be 0 besides initial state.
  ADvector constraints_lowerbound(n_constraints);
  ADvector constraints_upperbound(n_constraints);
  for (size_t idx = 0; idx < n_constraints; idx++) {
    constraints_lowerbound[idx] = 0;
    constraints_upperbound[idx] = 0;
  }
  constraints_lowerbound[x_start] = x;
  constraints_lowerbound[y_start] = y;
  constraints_lowerbound[psi_start] = psi;
  constraints_lowerbound[v_start] = v;
  constraints_lowerbound[cte_start] = cte;
  constraints_lowerbound[epsi_start] = epsi;

  constraints_upperbound[x_start] = x;
  constraints_upperbound[y_start] = y;
  constraints_upperbound[psi_start] = psi;
  constraints_upperbound[v_start] = v;
  constraints_upperbound[cte_start] = cte;
  constraints_upperbound[epsi_start] = epsi;
  
  // object that computes objective and constraints
  FG_eval fg_eval(coeffs);

  //
  // NOTE: You don't have to worry about these options
  //
  // options for IPOPT solver
  std::string options;
  // Uncomment this if you'd like more print information
  options += "Integer print_level  0\n";
  // NOTE: Setting sparse to true allows the solver to take advantage
  // of sparse routines, this makes the computation MUCH FASTER. If you
  // can uncomment 1 of these and see if it makes a difference or not but
  // if you uncomment both the computation time should go up in orders of
  // magnitude.
  options += "Sparse  true        forward\n";
  options += "Sparse  true        reverse\n";
  // NOTE: Currently the solver has a maximum time limit of 0.5 seconds.
  // Change this as you see fit.
  options += "Numeric max_cpu_time          0.5\n";

  // place to return solution
  CppAD::ipopt::solve_result<ADvector> solution;

  // solve the problem
  CppAD::ipopt::solve<ADvector, FG_eval>(
      options, vars, vars_lowerbound, vars_upperbound, constraints_lowerbound,
      constraints_upperbound, fg_eval, solution);

  // Check some of the solution values
  ok &= solution.status == CppAD::ipopt::solve_result<ADvector>::success;

  // Cost
  auto cost = solution.obj_value;
  std::cout << "Cost " << cost << std::endl;

  // TODO: Return the first actuator values. The variables can be accessed with
  // `solution.x[i]`.
  //
  // {...} is shorthand for creating a vector, so auto x1 = {1.0,2.0}
  // creates a 2 element double vector.
  
  vector<double> result;
  //std::cout << "return solution size : " << solution.x.size() << std::endl;
  for (size_t i = 0; i < N - 2 ; i++)
  {
    result.push_back(solution.x[delta_start + i]);
    result.push_back(solution.x[a_start + i]);
    result.push_back(solution.x[x_start + i]);
    result.push_back(solution.x[y_start + i]);
  }

  return result;
}
