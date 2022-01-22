#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  num_particles = 50;  // TODO: Set the number of particles
	Particle p;
  // generate random
  std::default_random_engine generator;
  std::normal_distribution<double> dist_x(0.0, std[0]);
  std::normal_distribution<double> dist_y(0.0, std[1]);
  std::normal_distribution<double> dist_theta(0.0, std[2]);
  // initialize with noise
  for (int i = 0; i < num_particles; i++){
    particles.push_back(p);
    particles[i].id = i;
    particles[i].x = x + dist_x(generator);
    particles[i].y = y + dist_y(generator);
    particles[i].theta = theta + dist_theta(generator);
    particles[i].weight = 1.0;
    weights.push_back(particles[i].weight);
    }
  std::cout << "Initialized!" << std::endl;
  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  std::default_random_engine generator;
  std::normal_distribution<double> dist_x(0.0, std_pos[0]);
  std::normal_distribution<double> dist_y(0.0, std_pos[1]);
  std::normal_distribution<double> dist_theta(0.0, std_pos[2]);
  
  for (int i = 0; i < num_particles; i++){
    // predict x, y and theta after delta_t
    if (fabs(yaw_rate) < 0.00001) {  
      particles[i].x += velocity * delta_t * cos(particles[i].theta) + dist_x(generator);
      particles[i].y += velocity * delta_t * sin(particles[i].theta) + dist_y(generator);
    } else {
      particles[i].x += (velocity / yaw_rate) * (sin(particles[i].theta + yaw_rate * delta_t) - sin(particles[i].theta)) + dist_x(generator);
      particles[i].y += (velocity / yaw_rate) * (-cos(particles[i].theta + yaw_rate * delta_t) + cos(particles[i].theta)) + dist_y(generator);
      particles[i].theta += (yaw_rate * delta_t) + dist_theta(generator);
    }
  }
}


void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {

    for (int s = 0; s < observations.size(); s++){
    double distance_error_min = 100000;
    LandmarkObs obs;
    // each predicted x, y (landmarks in global map)
    for (int k = 0; k < predicted.size(); k++){
      // calcurate distance between observed and predicted
      double d = dist(observations[s].x, observations[s].y, predicted[k].x, predicted[k].y);
      // if minimum
      if (d <= distance_error_min){
        // record ID
        observations[s].id = predicted[k].id;
        distance_error_min = d;
      } 
    }
  }

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  double weight_normalizer = 0.0;
  
  // for each Paticles
  for (int i = 0; i < num_particles; i++){
    // pick up landmarks in range
    vector<LandmarkObs> predicted;
    for (int j = 0; j < map_landmarks.landmark_list.size(); j++){
      // if landmark is in range
      //if (dist(particles[i].x, particles[i].y, map_landmarks.landmark_list[j].x_f, map_landmarks.landmark_list[j].y_f) <= sensor_range){
      if ( (fabs(particles[i].x - map_landmarks.landmark_list[j].x_f) <= sensor_range ) && ( fabs(particles[i].y - map_landmarks.landmark_list[j].y_f) <= sensor_range ) ){
        LandmarkObs pred;
        pred.id = map_landmarks.landmark_list[j].id_i;
        pred.x = map_landmarks.landmark_list[j].x_f;
        pred.y = map_landmarks.landmark_list[j].y_f;
        predicted.push_back(pred);
      }
    }
    
    // transform observed (measured) x, y from vehicle coordinate (map) to global coordinate (map)
    vector<LandmarkObs> observed;
    for (int k = 0; k < observations.size(); k++){
      LandmarkObs obs;
      // transform
      obs.x = particles[i].x + ( cos(particles[i].theta) * observations[k].x ) - ( sin(particles[i].theta) * observations[k].y );
      obs.y = particles[i].y + ( sin(particles[i].theta) * observations[k].x ) + ( cos(particles[i].theta) * observations[k].y );
      obs.id =  k;
      observed.push_back(obs);
    }
    
    // find closest landmark for each observed x, y 
    dataAssociation(predicted, observed); 
    
    particles[i].weight = 1.0;
    
    // wight update
    // for each observed x, y (they have been transformed to global map)
    for (int l = 0; l < observed.size(); l++){
      // for each predicted x,y 
      for (int m = 0; m < predicted.size(); m++){
        // use the result of dataAssociation (closest observed and predicted has same id)
        if ( predicted[m].id == observed[l].id){
          // update weight
          particles[i].weight = particles[i].weight * multiv_prob(std_landmark[0], std_landmark[1], observed[l].x, observed[l].y, predicted[m].x, predicted[m].y);
          //std::cout << "weight : " << multiv_prob(std_landmark[0], std_landmark[1], observed[l].x, observed[l].y, predicted[m].x, predicted[m].y) << std::endl;
        }
      }
    }
    //std::cout << "No." << i << " final weight : " << particles[i].weight << std::endl;
    weight_normalizer += particles[i].weight;
  }
  
  //std::cout << "weight normalizer : " << weight_normalizer << std::endl;
  
  // contain weights
  for (int i = 0; i < particles.size(); i++) {
    //particles[i].weight = particles[i].weight / weight_normalizer;
    weights[i] = particles[i].weight;
  }
}

double ParticleFilter::multiv_prob(double sig_x, double sig_y, double x_obs, double y_obs, double mu_x, double mu_y) {
  // calculate normalization term
  double gauss_norm;
  gauss_norm = 1 / (2 * M_PI * sig_x * sig_y);

  // calculate exponent
  double exponent;
  exponent = (pow(x_obs - mu_x, 2) / (2 * pow(sig_x, 2)))
               + (pow(y_obs - mu_y, 2) / (2 * pow(sig_y, 2)));
    
  // calculate weight using normalization terms and exponent
  double weight;
  weight = gauss_norm * exp(-exponent);
    
  return weight;
}

void ParticleFilter::resample() {

  vector<Particle> new_p;
  std::default_random_engine generator;
  std::uniform_int_distribution<int> particle_index(0, num_particles - 1);
  int index = particle_index(generator);
  double beta = 0.0;
  double mw = -1.0;
  
    for (int i = 0; i < num_particles; i++){
      if (weights[i] > mw){
        mw = weights[i];
      }
    }
  
    for (int j = 0; j < num_particles; j++){
      std::uniform_real_distribution<double> random_weight(0.0, 2*mw);
      beta += random_weight(generator);

      while ( beta > weights[index] ){
        beta -= weights[index];
        index = (index + 1) % num_particles;
      }
      new_p.push_back(particles[index]);
    }
    particles = new_p;
  //std::cout << "Resampled! ( N = " << particles.size() << ")" << std::endl;
}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}
