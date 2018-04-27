#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

#define MIN_NUMBER 0.0001
/**
 * Initializes Unscented Kalman filter
 * This is scaffolding, do not modify
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(5);

  // initial covariance matrix
  P_ = MatrixXd(5, 5);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 10.5;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 0.35;
  
  //DO NOT MODIFY measurement noise values below these are provided by the sensor manufacturer.
  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;
  //DO NOT MODIFY measurement noise values above these are provided by the sensor manufacturer.
  
  //set state dimension
  n_x_ = 5;

  //set augmented dimension
  n_aug_ = 7;
  
  //define spreading parameter
  lambda_ = 3 - n_aug_;

  //set measurement dimension, radar can measure r, phi, and r_dot
  radar_n_z_ = 3;
  
  //set measurement dimension, laser can measure px, py
  laser_n_z_ = 2;
  
  //create sigma point matrix
  Xsig_aug_ = MatrixXd(n_aug_, 2 * n_aug_ + 1);
  
  //create matrix with predicted sigma points as columns
  Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);
  
  //create matrix for sigma points in measurement space
  radar_Zsig_pred_ = MatrixXd(radar_n_z_, 2 * n_aug_ + 1);
  
  //create matrix for sigma points in measurement space
  laser_Zsig_pred_ = MatrixXd(laser_n_z_, 2 * n_aug_ + 1);

  //measurement predict value
  radar_z_pred_ = VectorXd(radar_n_z_);

  //measurement predict value  
  laser_z_pred_ = VectorXd(laser_n_z_);
  
  ///* measurement covariance matrix
  radar_S_ = MatrixXd(radar_n_z_,radar_n_z_);
  
  ///* measurement covariance matrix
  laser_S_ = MatrixXd(laser_n_z_,laser_n_z_);
  
  //set vector for weights_
  weights_ = VectorXd(2*n_aug_+1);
  
  // set weights
  double weight_0 = lambda_/(lambda_+n_aug_);
  weights_(0) = weight_0;
  for (int i=1; i<2*n_aug_+1; i++) {  //2n+1 weights_
    double weight = 0.5/(n_aug_+lambda_);
    weights_(i) = weight;
  }

  // lidar NIS
  NIS_laser_ = 0.0;

  // radar NIS
  NIS_radar_ = 0.0;

  // Lidar NIS file
  laser_NIS_file_.open("laser_nis_file.txt");

  ///* Radar NIS file
  radar_NIS_file_.open("radar_nis_file.txt");
}

UKF::~UKF() 
{
  // Lidar NIS file
  laser_NIS_file_.close();

  ///* Radar NIS file
  radar_NIS_file_.close();
}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {

  if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
    //std::cout << "---------MeasurementPackage::RADAR----------" << std::endl;
    if(!use_radar_)
      return;
  }
  else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
    //std::cout << "---------MeasurementPackage::LASER----------" << std::endl;
    if(!use_laser_)
      return;
  }
    
  /*****************************************************************************
   *  Initialization
   ****************************************************************************/
  if (!is_initialized_) {
  
    // first measurement
    std::cout << "UKF initialize " << std::endl;

    float px = 0.0;
    float py = 0.0;
    float v = 0.0;
    float yaw = 0.0;
    float yawd = 0.0;   

    if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
      /**
      Convert radar from polar to cartesian coordinates and initialize state.
      */
      float rho = meas_package.raw_measurements_[0];
      float phi = meas_package.raw_measurements_[1];
      float rho_dot = meas_package.raw_measurements_[2]; 
      px = rho * cos(phi);
      py = rho * sin(phi);
      v = rho_dot;
      yaw = 0.0;
      yawd = 0.0;
      
      x_ << px, py, v, yaw, yawd;   
      
    }
    else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
      /**
      Initialize state.
      */
      px = meas_package.raw_measurements_[0];
      py = meas_package.raw_measurements_[1];
      v = 0.0;
      yaw = 0.0;
      yawd = 0.0; 
      
      x_ << px, py, v, yaw, yawd;
      
    }
    
	//the initial state covariance matrix P_        
	P_ <<  0.0043,   -0.0013,    0.0030,   -0.0022,   -0.0020,
          -0.0013,    0.0077,    0.0011,    0.0071,    0.0060,
           0.0030,    0.0011,    0.0054,    0.0007,    0.0008,
          -0.0022,    0.0071,    0.0007,    0.0098,    0.0100,
          -0.0020,    0.0060,    0.0008,    0.0100,    0.0123;   
    //update time
    time_us_ = meas_package.timestamp_;
    
    // done initializing, no need to predict or update
    is_initialized_ = true;
    
    //std::cout << "UKF succeed initialize " << std::endl;
    
    return;
  }

  /*****************************************************************************
   *  Prediction
   ****************************************************************************/

	//compute the time elapsed between the current and previous measurements
	float delta_t = (meas_package.timestamp_ - time_us_) / 1000000.0;	//dt - expressed in seconds
	time_us_ = meas_package.timestamp_;

	//predict		   
  Prediction(delta_t);

  /*****************************************************************************
   *  Update
   ****************************************************************************/

  if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
    // Radar updates
    UpdateRadar(meas_package);
  } else {
    // Laser updates
    UpdateLidar(meas_package);
  }

  // print the output
  //std::cout << "x_ = " << std::endl << x_ << std::endl;
  //std::cout << "P_ = " << std::endl << P_ << std::endl;
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
  /**
  TODO:

  Complete this function! Estimate the object's location. Modify the state
  vector, x_. Predict sigma points, the state, and the state covariance matrix.
  */
  GenerateSigmaPoints();
  SigmaPointPrediction(delta_t);
  PredictMeanAndCovariance();
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Use lidar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the lidar NIS.
  */
  PredictLaserMeasurement();
  LaserUpdateState(meas_package.raw_measurements_);


}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Use radar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the radar NIS.
  */
  PredictRadarMeasurement();
  RadarUpdateState(meas_package.raw_measurements_);

}

void UKF::GenerateSigmaPoints() {

  //create augmented mean vector
  VectorXd x_aug = VectorXd(7);

  //create augmented state covariance
  MatrixXd P_aug = MatrixXd(7, 7);

  //create augmented mean state
  x_aug.head(5) = x_;
  x_aug(5) = 0;
  x_aug(6) = 0;

  //create augmented covariance matrix
  P_aug.fill(0.0);
  P_aug.topLeftCorner(5,5) = P_;
  P_aug(5,5) = std_a_*std_a_;
  P_aug(6,6) = std_yawdd_*std_yawdd_;

  //create square root matrix
  MatrixXd L = P_aug.llt().matrixL();

  //create augmented sigma points
  Xsig_aug_.col(0)  = x_aug;
  for (int i = 0; i< n_aug_; i++)
  {
    Xsig_aug_.col(i+1)       = x_aug + sqrt(lambda_+n_aug_) * L.col(i);
    Xsig_aug_.col(i+1+n_aug_) = x_aug - sqrt(lambda_+n_aug_) * L.col(i);
  }
  
  //print result
  //std::cout << "Xsig_aug_ = " << std::endl << Xsig_aug_ << std::endl;

}

void UKF::SigmaPointPrediction(double delta_t) {

  //std::cout << "delta_t = "  << delta_t << std::endl;
  
  //predict sigma points
  for (int i = 0; i< 2*n_aug_+1; i++)
  {
    //extract values for better readability
    double p_x = Xsig_aug_(0,i);
    double p_y = Xsig_aug_(1,i);
    double v = Xsig_aug_(2,i);
    double yaw = Xsig_aug_(3,i);
    double yawd = Xsig_aug_(4,i);
    double nu_a = Xsig_aug_(5,i);
    double nu_yawdd = Xsig_aug_(6,i);

    //predicted state values
    double px_p, py_p;

    //avoid division by zero
    if (fabs(yawd) > 0.001) {
        px_p = p_x + v/yawd * ( sin (yaw + yawd*delta_t) - sin(yaw));
        py_p = p_y + v/yawd * ( cos(yaw) - cos(yaw+yawd*delta_t) );
    }
    else {
        px_p = p_x + v*delta_t*cos(yaw);
        py_p = p_y + v*delta_t*sin(yaw);
    }

    double v_p = v;
    double yaw_p = yaw + yawd*delta_t;
    double yawd_p = yawd;

    //add noise
    px_p = px_p + 0.5*nu_a*delta_t*delta_t * cos(yaw);
    py_p = py_p + 0.5*nu_a*delta_t*delta_t * sin(yaw);
    v_p = v_p + nu_a*delta_t;

    yaw_p = yaw_p + 0.5*nu_yawdd*delta_t*delta_t;
    yawd_p = yawd_p + nu_yawdd*delta_t;

    //write predicted sigma point into right column
    Xsig_pred_(0,i) = px_p;
    Xsig_pred_(1,i) = py_p;
    Xsig_pred_(2,i) = v_p;
    Xsig_pred_(3,i) = yaw_p;
    Xsig_pred_(4,i) = yawd_p;
  }

  //print result
  //std::cout << "Xsig_pred_ = " << std::endl << Xsig_pred_ << std::endl;

}

void UKF::PredictMeanAndCovariance() {

  //predicted state mean
  x_.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //iterate over sigma points
    x_ = x_ + weights_(i) * Xsig_pred_.col(i);
  }

  //predicted state covariance matrix
  P_.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //iterate over sigma points

    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    //angle normalization
    while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
    while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;

    P_ = P_ + weights_(i) * x_diff * x_diff.transpose() ;
  }

  //print result
  //std::cout << "Predicted state" << std::endl;
  //std::cout << x_ << std::endl;
  //std::cout << "Predicted covariance matrix" << std::endl;
  //std::cout << P_ << std::endl;

}

void UKF::PredictRadarMeasurement() {

  //transform sigma points into measurement space
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points

    // extract values for better readibility
    double p_x = Xsig_pred_(0,i);
    double p_y = Xsig_pred_(1,i);
    double v  = Xsig_pred_(2,i);
    double yaw = Xsig_pred_(3,i);

    double v1 = cos(yaw)*v;
    double v2 = sin(yaw)*v;
    
    if (fabs(p_x) < MIN_NUMBER) {
      std::cout << "PredictRadarMeasurement() - Error - p_x too small, px = " << p_x << std::endl;
      p_x = MIN_NUMBER;
    }

    float rho = sqrt(p_x * p_x + p_y * p_y);
    if (fabs(rho) < MIN_NUMBER) {
      std::cout << "PredictRadarMeasurement() - Error - rho too small, rho = " << rho << std::endl;
      rho = MIN_NUMBER;
    }

    // measurement model
    radar_Zsig_pred_(0,i) = rho;                        //rho
    radar_Zsig_pred_(1,i) = atan2(p_y,p_x);             //phi
    radar_Zsig_pred_(2,i) = (p_x*v1 + p_y*v2 ) / rho ;  //r_dot
  }

  //mean predicted measurement
  radar_z_pred_.fill(0.0);
  for (int i=0; i < 2*n_aug_+1; i++) {
      radar_z_pred_ = radar_z_pred_ + weights_(i) * radar_Zsig_pred_.col(i);
  }

  radar_S_.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points
    //residual
    VectorXd z_diff = radar_Zsig_pred_.col(i) - radar_z_pred_;

    //angle normalization
    while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
    while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

    radar_S_ = radar_S_ + weights_(i) * z_diff * z_diff.transpose();
  }

  //add measurement noise covariance matrix
  MatrixXd R = MatrixXd(radar_n_z_, radar_n_z_);
  R <<    std_radr_*std_radr_, 0, 0,
          0, std_radphi_*std_radphi_, 0,
          0, 0,std_radrd_*std_radrd_;
  radar_S_ = radar_S_ + R;


  //print result
  //std::cout << "radar_z_pred_: " << std::endl << radar_z_pred_ << std::endl;
  //std::cout << "radar_S_: " << std::endl << radar_S_ << std::endl;

}

void UKF::RadarUpdateState(const VectorXd &z) {

  //create matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd(n_x_, radar_n_z_);

  //calculate cross correlation matrix
  Tc.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points

    //residual
    VectorXd z_diff = radar_Zsig_pred_.col(i) - radar_z_pred_;
    //angle normalization
    while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
    while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    //angle normalization
    while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
    while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;

    Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
  }

  //Kalman gain K;
  MatrixXd K = Tc * radar_S_.inverse();

  //residual
  VectorXd z_diff = z - radar_z_pred_;

  //angle normalization
  while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
  while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

  //update state mean and covariance matrix
  x_ = x_ + K * z_diff;
  P_ = P_ - K*radar_S_*K.transpose();
  
  // Compute laser NIS value
  NIS_radar_ = z_diff.transpose() * radar_S_.inverse() * z_diff;
  // Write laser NIS value to file
  radar_NIS_file_ << NIS_radar_ << std::endl;
    
  //print result
  //std::cout << "Updated radar state x: " << std::endl << x_ << std::endl;
  //std::cout << "Updated radar state covariance P: " << std::endl << P_ << std::endl;

}


void UKF::PredictLaserMeasurement() {

  //transform sigma points into measurement space
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points

    // extract values for better readibility
    double p_x = Xsig_pred_(0,i);
    double p_y = Xsig_pred_(1,i);
    double v  = Xsig_pred_(2,i);
    double yaw = Xsig_pred_(3,i);

    // measurement model
    laser_Zsig_pred_(0,i) = p_x;       
    laser_Zsig_pred_(1,i) = p_y;      
  }

  //mean predicted measurement
  laser_z_pred_.fill(0.0);
  for (int i=0; i < 2*n_aug_+1; i++) {
      laser_z_pred_ = laser_z_pred_ + weights_(i) * laser_Zsig_pred_.col(i);
  }

  laser_S_.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points
    //residual
    VectorXd z_diff = laser_Zsig_pred_.col(i) - laser_z_pred_;

    laser_S_ = laser_S_ + weights_(i) * z_diff * z_diff.transpose();
  }

  //add measurement noise covariance matrix
  MatrixXd R = MatrixXd(laser_n_z_, laser_n_z_);
  R <<    std_laspx_*std_laspx_, 0,
          0, std_laspy_*std_laspy_;

  laser_S_ = laser_S_ + R;

  //print result
  //std::cout << "laser_z_pred_: " << std::endl << laser_z_pred_ << std::endl;
  //std::cout << "laser_S_: " << std::endl << laser_S_ << std::endl;

}

void UKF::LaserUpdateState(const VectorXd &z) {

  //create matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd(n_x_, laser_n_z_);

  //calculate cross correlation matrix
  Tc.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points

    //residual
    VectorXd z_diff = laser_Zsig_pred_.col(i) - laser_z_pred_;

    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    //angle normalization
    while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
    while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;

    Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
  }

  //Kalman gain K;
  MatrixXd K = Tc * laser_S_.inverse();

  //residual
  VectorXd z_diff = z - laser_z_pred_;

  //update state mean and covariance matrix
  x_ = x_ + K * z_diff;
  P_ = P_ - K* laser_S_ * K.transpose();
  
  // Compute laser NIS value
  NIS_laser_ = z_diff.transpose() * laser_S_.inverse() * z_diff;
  // Write laser NIS value to file
  laser_NIS_file_ << NIS_laser_ << std::endl;
  
  //print result
  //std::cout << "Updated laser state x: " << std::endl << x_ << std::endl;
  //std::cout << "Updated laser state covariance P: " << std::endl << P_ << std::endl;

}
