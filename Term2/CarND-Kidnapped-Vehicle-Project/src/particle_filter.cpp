/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double gps_x, double gps_y, double gps_theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	
  //std::cout << "PF start initialize " << std::endl;
  
	// Number of particles to draw
	num_particles = 100; 
	
	//Set standard deviations for x, y, and theta
  double std_x = std[0];
  double std_y = std[1];
  double std_theta = std[2];

	// Create normal distributions for x, y and theta
	default_random_engine gen;
  normal_distribution<double> dist_x(gps_x, std_x);
	normal_distribution<double> dist_y(gps_y, std_y);
	normal_distribution<double> dist_theta(gps_theta, std_theta);

	for (int i = 0; i < num_particles; i++) {
		Particle particle;
		// Sample  and from these normal distrubtions 
    particle.id = i;
    particle.x = dist_x(gen);
    particle.y = dist_y(gen);
    particle.theta = dist_theta(gen);	
    particle.weight = 1.0;
    particles.push_back(particle);
    weights.push_back(1.0);
	}
	
	// filter is initialized
	is_initialized = true;
  
  //std::cout << "PF succeed initialize " << std::endl;
  
  return;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	//Set standard deviations for x, y, and theta
  double std_x = std_pos[0];
  double std_y = std_pos[1];
  double std_theta = std_pos[2];

	// Create normal distributions for x, y and theta
	default_random_engine gen;
  normal_distribution<double> dist_x(0, std_x*delta_t);
	normal_distribution<double> dist_y(0, std_y*delta_t);
	normal_distribution<double> dist_theta(0, std_theta*delta_t);
	
  // Predict the position of ech particle using the CTRV motion model
  for (int i = 0; i < num_particles; i++)
  {
    // if yaw_rate == 0 the trajectory is a line 
    if (fabs(yaw_rate) < 0.001)
    {
      particles[i].x += velocity * delta_t * cos(particles[i].theta);
      particles[i].y += velocity * delta_t * sin(particles[i].theta);
      particles[i].theta += yaw_rate * delta_t;
    }
    else // yaw rate != 0
    {
      particles[i].x += (velocity/yaw_rate) * (sin(particles[i].theta + yaw_rate * delta_t) - sin(particles[i].theta));
      particles[i].y += (velocity/yaw_rate) * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate * delta_t));
      particles[i].theta += yaw_rate * delta_t;
    }

    // Add random Gaussian noise to each particle to account
    particles[i].x += dist_x(gen);
    particles[i].y += dist_y(gen);
    particles[i].theta += dist_theta(gen);
    
    //std::cout << "predicted particles x : " << particles[i].x << std::endl;
    //std::cout << "predicted particles y : " << particles[i].y << std::endl;
    //std::cout << "predicted particles theta : " << particles[i].theta << std::endl;
    
  }

  return;
  
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
  
  int nearest_predicttion = 0;
  double nearest_distance = 0;
  double delta_x = 0;
  double delta_y = 0;
  double delta = 0;
  
  // Iterate over all observed measurements  
  for (int i = 0; i < observations.size(); ++i)
  {
    delta_x = predicted[0].x - observations[i].x;
    delta_y = predicted[0].y - observations[i].y;
    delta = delta_x * delta_x + delta_y * delta_y;
    nearest_distance = delta;
    nearest_predicttion = 0;
    // Iterate over all predicted measurements find out the closest measurement
    for (int j = 1; j < predicted.size(); ++j)
    {
      delta_x = predicted[j].x - observations[i].x;
      delta_y = predicted[j].y - observations[i].y;
      delta = delta_x * delta_x + delta_y * delta_y;
      if (delta < nearest_distance)
      {
        nearest_distance = delta;
        nearest_predicttion = j;
      }
    }
    observations[i].id = nearest_predicttion;
  }
  
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
	
	// landmark observations standard deviations
  double stdx = std_landmark[0];
  double stdy = std_landmark[1];
  
  double gauss_norm_x = 1.0 / (2.0 * stdx * stdx);
  double gauss_norm_y = 1.0 / (2.0 * stdy * stdy);
  double gauss_norm_xy = 1.0 / (2.0 * M_PI * stdx * stdy);
  
  // Iterate over all particles to update each particle's weight
  for (int i = 0; i < num_particles; i++)
  {
    // vector of observations in map coordinates
    std::vector<LandmarkObs> observations_in_map;
    
    double particle_x = particles[i].x;
    double particle_y = particles[i].y;
    double particle_theta = particles[i].theta;

    // filter map landmarks in range of sensor from the current car pos
    std::vector<LandmarkObs> landmarks_in_range;

    // Find landmarks which are in sensor range
    for (int j = 0;  j < map_landmarks.landmark_list.size(); j++)
    {
      int landmark_id = map_landmarks.landmark_list[j].id_i;
      double landmark_x = map_landmarks.landmark_list[j].x_f;
      double landmark_y = map_landmarks.landmark_list[j].y_f;
      double delta_x = landmark_x - particle_x;
      double delta_y = landmark_y - particle_y;
      double distance = sqrt(delta_x * delta_x + delta_y * delta_y);
      if (distance < sensor_range)
      {
        LandmarkObs landmark = {landmark_id, landmark_x, landmark_y};
        landmarks_in_range.push_back(landmark);
      }
    }
    
    // transform observations from vehicle coordinates to map coordinates
    for (int j = 0; j < observations.size(); ++j)
    {
      int obs_id = observations[j].id;
      double map_x = particle_x + observations[j].x * cos(particle_theta) - observations[j].y * sin(particle_theta);
      double map_y = particle_y + observations[j].y * cos(particle_theta) + observations[j].x * sin(particle_theta);

      LandmarkObs map_obs = {obs_id, map_x, map_y};
      observations_in_map.push_back(map_obs);
    }

    // Associate landmarks in range of sensor to car observations on the map
    dataAssociation(landmarks_in_range, observations_in_map);

    // compute weight 
    double particle_weight = 1.0;

    for (int j = 0; j < observations_in_map.size(); ++j)
    {
      int id = observations_in_map[j].id;
      double predict_x = observations_in_map[j].x;
      double predict_y = observations_in_map[j].y;

      double assoc_x = landmarks_in_range[id].x;
      double assoc_y = landmarks_in_range[id].y;

      double dx = predict_x - assoc_x;
      double dy = predict_y - assoc_y;
      
      //calculate exponent
      double exponent= gauss_norm_x * dx * dx + gauss_norm_y *dy * dy;
      //calculate weight using normalization terms and exponent
      double weight= gauss_norm_xy * exp(-exponent);

      particle_weight *= weight;
    }
    
    // add the weight to the list of weights
    weights[i] = particle_weight;
    // update the particle weight
    particles[i].weight = particle_weight;
  }
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	
	// particle array
  vector<Particle> new_particles;
  
  // random_engine  
	default_random_engine gen;

	// create a discrete distribution with weights
  discrete_distribution<> dist_weights(weights.begin(), weights.end());

  // resample particles
  for (int i = 0; i < num_particles; ++i)
      new_particles.push_back(particles[dist_weights(gen)]);

	particles = new_particles;
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates
  
    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
