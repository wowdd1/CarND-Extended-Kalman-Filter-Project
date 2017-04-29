#include "kalman_filter.h"
#include <iostream>

using Eigen::MatrixXd;
using Eigen::VectorXd;
using namespace std;

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}


MatrixXd GetRadarMeas(const VectorXd& x_state);


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
  // predict the state
  x_ = F_ * x_;
  MatrixXd Ft = F_.transpose();
  P_ = F_ * P_ * Ft + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {
  // update the state by using Kalman Filter equations
  VectorXd z_pred = H_ * x_;
  VectorXd y = z - z_pred;
  MatrixXd Ht = H_.transpose();
  MatrixXd S = H_ * P_ * Ht + R_;
  MatrixXd Si = S.inverse();
  MatrixXd PHt = P_ * Ht;
  MatrixXd K = PHt * Si;

  // new estimate
  x_ = x_ + (K * y);
  long x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(x_size, x_size);
  P_ = (I - K * H_) * P_;
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  // update the state by using Extended Kalman Filter equations. ie. use h(x') to calculate y and Hj_ to calculate S, K and P

  // convert the x_ vector from cartesian to polar coords so that it can be compared against raw_measurements
  float px = x_[0];
  float py = x_[1];
  float vx = x_[2];
  float vy = x_[3];

  // accomodate for divide by 0 error (same as in Hj calculation in tools.cpp)
  float c1 = px*px + py*py;
  // if division by zero, set denominator to a small number
  while(fabs(c1) < 0.0001){
    cout << "ro_dot - Error - Division by Zero" << endl;
    cout << "px=" << px << " and py=" << py << " ... adding 0.001 and continuing" << endl;
    px += 0.001;
    py += 0.001;
    c1 = px*px + py*py;
  }

  float ro = sqrt(px*px + py*py);
  float phi = atan2(py, px);
  float ro_dot = (px*vx + py*vy)/ro;                //Caution, potential divide by 0 error....
  VectorXd z_pred_polar = VectorXd(3);
  z_pred_polar << ro, phi, ro_dot;

  // the Jacobian matrix is fed into this function as H_
  MatrixXd Hj_ = H_;
  MatrixXd Hjt = Hj_.transpose();
  // x_ is converted to cartesian coordinates in FusionEKF.cpp
  VectorXd y = z - z_pred_polar;

  if (y[1] > M_PI) {
    y[1] = y[1] - 2*M_PI;
  }
  else if (y[1] < -M_PI) {
    y[1] = y[1] + 2 * M_PI;
  }
  else {
    //do nothing keep it as it is
  }

  MatrixXd S = Hj_ * P_ * Hjt + R_;
  MatrixXd Si = S.inverse();
  MatrixXd PHjt = P_ * Hjt;
  MatrixXd K = PHjt * Si;

  // new estimate
  x_ = x_ + (K * y);
  long x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(x_size, x_size);
  P_ = (I - K * Hj_) * P_;
}