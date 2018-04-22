#include <iostream>
#include "kalman_filter.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

// Please note that the Eigen library does not initialize 
// VectorXd or MatrixXd objects with zeros upon creation.

#define MIN_NUMBER 0.0001

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
}

void KalmanFilter::Predict() {
  x_ = F_ * x_;
  MatrixXd Ft = F_.transpose();
  P_ = F_ * P_ * Ft + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {
  //update with measurement of linear
  VectorXd z_pred = H_ * x_;
  VectorXd y = z - z_pred;
  MatrixXd Ht = H_.transpose(); 
  MatrixXd S = H_ * P_ * Ht + R_;
  MatrixXd Si = S.inverse();
  MatrixXd PHt = P_ * Ht;
  MatrixXd K = PHt * Si;

  //new estimate
  x_ = x_ + (K * y);
  long x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(x_size, x_size);
  P_ = (I - K * H_) * P_;
  
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  //update with measurement of nolinear
  float px = x_(0);
  float py = x_(1);
  float vx = x_(2);
  float vy = x_(3);

  float rho = 0.0;
  float phi = 0.0;
  float rho_dot = 0.0;
  
  VectorXd z_pred = VectorXd(3);
  
  // avoid devide by zero
  if (fabs(px) < MIN_NUMBER) {
    std::cout << "UpdateEKF() - Error - px too small, px = " << px << std::endl;
    px = MIN_NUMBER;
  }

  rho = sqrt(px * px + py * py);
  // avoid devide by zero
  if (fabs(rho) < MIN_NUMBER) {
    std::cout << "UpdateEKF() - Error - rho too small, rho = " << rho << std::endl;
    rho = MIN_NUMBER;
  }

  phi = atan2(py, px);
  
  rho_dot = (px * vx + py * vy) / rho;

  z_pred << rho, phi, rho_dot;
  VectorXd y = z - z_pred;
  // normalling angles
  while( y[1] > M_PI)
    y[1] -= 2*M_PI;
  while( y[1] < -M_PI)
    y[1] += 2*M_PI;
    
  MatrixXd Ht = H_.transpose();
  MatrixXd S = H_ * P_ * Ht + R_;
  MatrixXd Si = S.inverse();
  MatrixXd PHt = P_ * Ht;
  MatrixXd K = PHt * Si;

  //new estimate
  x_ = x_ + (K * y);
  long x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(x_size, x_size);
  P_ = (I - K * H_) * P_;
  
}
