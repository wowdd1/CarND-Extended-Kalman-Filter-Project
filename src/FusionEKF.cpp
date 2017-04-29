#include "FusionEKF.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/*
 * Constructor.
 */
FusionEKF::FusionEKF() {
  is_initialized_ = false;

  previous_timestamp_ = 0;

  // initializing matrices
  R_laser_ = MatrixXd(2, 2);
  R_radar_ = MatrixXd(3, 3);
  H_laser_ = MatrixXd(2, 4);
  Hj_ = MatrixXd(3, 4);
  F_ = MatrixXd(4, 4);
  P_ = MatrixXd(4, 4);
  Q_ = MatrixXd(4, 4);

  // measurement covariance matrix - laser
  R_laser_ << 0.0225, 0,
              0, 0.0225;

  // measurement covariance matrix - radar
  R_radar_ << 0.09, 0, 0,
              0, 0.0009, 0,
              0, 0, 0.09;

  // measurement function matrix - laser
  H_laser_ << 1, 0, 0, 0,
              0, 1, 0, 0;

  // measurement function Jacobian matrix - radar
  // the Jacobian is used to convert a non-linear function
  // to it's approximate linear equivalent
  VectorXd x_state = VectorXd(4);
  Hj_ = tools.CalculateJacobian(x_state);

  // state transition matrix
  F_ << 1, 0, 1, 0,
        0, 1, 0, 1,
        0, 0, 1, 0,
        0, 0, 0, 1;

  // state covariance matrix
  P_ << 1000,    0, 0   , 0,
        0   , 1000, 0   , 0,
        0   ,    0, 1000, 0,
        0   ,    0, 0   , 1000;

  // process covariance matrix
  Q_ << 0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0;
}

/**
* Destructor.
*/
FusionEKF::~FusionEKF() {}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {


  /*****************************************************************************
   *  Initialization
   ****************************************************************************/
  if (!is_initialized_) {
    // first measurement
    cout << "EKF: " << endl;
    ekf_.x_ = VectorXd(4);
    ekf_.x_ << 1, 1, 1, 1;

    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
      /**
      Convert radar from polar to cartesian coordinates and initialize state.
      */
      // convert from polar to cartesian coords
      float ro = measurement_pack.raw_measurements_[0];
      float phi = measurement_pack.raw_measurements_[1];
      float ro_dot = measurement_pack.raw_measurements_[2];
      float px = ro * cos(phi);
      float py = ro * sin(phi);
      float vx = 0;
      float vy = 0;
      ekf_.x_ << px, py, vx, vy;
      // initialise the extended kalman filter matrices
      ekf_.Init(ekf_.x_, P_, F_, Hj_, R_radar_, Q_);
      // initialise the previous timestamp
      previous_timestamp_ = measurement_pack.timestamp_;
    }
    else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
      /**
      Initialize state.
      */
      // set the state with the initial location and zero velocity
      ekf_.x_ << measurement_pack.raw_measurements_[0], measurement_pack.raw_measurements_[1], 0, 0;
      // initialise the extended kalman filter matrices
      ekf_.Init(ekf_.x_, P_, F_, H_laser_, R_laser_, Q_);
      // initialise the previous timestamp
      previous_timestamp_ = measurement_pack.timestamp_;
    }
    // done initializing, no need to predict or update
    is_initialized_ = true;
    return;
  }

  /*****************************************************************************
   *  Prediction
   ****************************************************************************/
  // compute the time elapsed between the current and previous measurements
  float dt = (measurement_pack.timestamp_ - previous_timestamp_) / 1000000.0;	//dt - expressed in seconds
  previous_timestamp_ = measurement_pack.timestamp_;

  // process noise parameters
  int noise_ax = 9;
  int noise_ay = 9;
  // update the state transition matrix F according to the new elapsed time
  ekf_.F_(0, 2) = dt;
  ekf_.F_(1, 3) = dt;
  float dt_2 = pow(dt, 2);
  float dt_3 = pow(dt, 3);
  float dt_4 = pow(dt, 4);
  // update the process noise covariance matrix
  ekf_.Q_ << dt_4/4*noise_ax, 0, dt_3/2*noise_ax, 0,
             0, dt_4/4*noise_ay, 0, dt_3/2*noise_ay,
             dt_3/2*noise_ax, 0, dt_2*noise_ax, 0,
             0, dt_3/2*noise_ay, 0, dt_2*noise_ay;

  ekf_.Predict();

  /*****************************************************************************
   *  Update
   ****************************************************************************/
  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
    // Radar updates
    Hj_ = tools.CalculateJacobian(ekf_.x_);
    ekf_.H_ = Hj_;
    ekf_.R_ = R_radar_;
    ekf_.UpdateEKF(measurement_pack.raw_measurements_);
  } 
  else {
    // Laser updates
    ekf_.H_ = H_laser_;
    ekf_.R_ = R_laser_;
    ekf_.Update(measurement_pack.raw_measurements_);
  }

  // print the output
  cout << "x_ = " << ekf_.x_ << endl;
  cout << "P_ = " << ekf_.P_ << endl;
}
