#ifndef UKF_H
#define UKF_H

#include "measurement_package.h"
#include "Eigen/Dense"
#include <vector>
#include <string>
#include <fstream>

using Eigen::MatrixXd;
using Eigen::VectorXd;

class UKF {
public:

  ///* initially set to false, set to true in first call of ProcessMeasurement
  bool is_initialized_;

  ///* if this is false, laser measurements will be ignored (except for init)
  bool use_laser_;

  ///* if this is false, radar measurements will be ignored (except for init)
  bool use_radar_;

  ///* state vector: [pos1 pos2 vel_abs yaw_angle yaw_rate] in SI units and rad
  VectorXd x_;

  ///* state covariance matrix
  MatrixXd P_;

  ///* predicted sigma points matrix
  MatrixXd Xsig_pred_;
  
  ///* augment sigma points matrix
  MatrixXd Xsig_aug_;
  
  ///* predicted sigma points matrix
  MatrixXd radar_Zsig_pred_;
  
  ///* predicted sigma points matrix
  MatrixXd laser_Zsig_pred_;
  
  //measurement predict value
  VectorXd radar_z_pred_;
  
  //measurement predict value
  VectorXd laser_z_pred_;
  
  ///* measurement covariance matrix
  MatrixXd radar_S_;

  ///* measurement covariance matrix
  MatrixXd laser_S_;
  
  ///* time when the state is true, in us
  long long time_us_;

  ///* Process noise standard deviation longitudinal acceleration in m/s^2
  double std_a_;

  ///* Process noise standard deviation yaw acceleration in rad/s^2
  double std_yawdd_;

  ///* Laser measurement noise standard deviation position1 in m
  double std_laspx_;

  ///* Laser measurement noise standard deviation position2 in m
  double std_laspy_;

  ///* Radar measurement noise standard deviation radius in m
  double std_radr_;

  ///* Radar measurement noise standard deviation angle in rad
  double std_radphi_;

  ///* Radar measurement noise standard deviation radius change in m/s
  double std_radrd_ ;

  ///* Weights of sigma points
  VectorXd weights_;

  ///* State dimension
  int n_x_;

  ///* Augmented state dimension
  int n_aug_;

  ///* Sigma point spreading parameter
  double lambda_;

  // measurement dimension
  int radar_n_z_;
  
  // measurement dimension
  int laser_n_z_;
  
  // lidar NIS
  double NIS_laser_;

  // radar NIS
  double NIS_radar_;

  // Lidar NIS file
  std::ofstream laser_NIS_file_;

  ///* Radar NIS file
  std::ofstream radar_NIS_file_;
  
  /**
   * Constructor
   */
  UKF();

  /**
   * Destructor
   */
  virtual ~UKF();

  /**
   * ProcessMeasurement
   * @param meas_package The latest measurement data of either radar or laser
   */
  void ProcessMeasurement(MeasurementPackage meas_package);

  /**
   * Prediction Predicts sigma points, the state, and the state covariance
   * matrix
   * @param delta_t Time between k and k+1 in s
   */
  void Prediction(double delta_t);

  /**
   * Updates the state and the state covariance matrix using a laser measurement
   * @param meas_package The measurement at k+1
   */
  void UpdateLidar(MeasurementPackage meas_package);

  /**
   * Updates the state and the state covariance matrix using a radar measurement
   * @param meas_package The measurement at k+1
   */
  void UpdateRadar(MeasurementPackage meas_package);
  
  void GenerateSigmaPoints();
  
  void SigmaPointPrediction(double delta_t);
  
  void PredictMeanAndCovariance();
  
  void PredictRadarMeasurement();
  
  void RadarUpdateState(const VectorXd &z);
  
  void PredictLaserMeasurement();
  
  void LaserUpdateState(const VectorXd &z);
};

#endif /* UKF_H */
