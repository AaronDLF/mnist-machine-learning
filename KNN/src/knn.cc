#include "knn.hpp"
#include <cmath>
#include <limits>
#include <map>
#include "stdint.h"
#include "data_handler.hpp"


knn::knn(int val)
{
  k = val;
}

knn::knn(){
  //Nothing inside
}

knn::~knn(){
  //Nothing inside
}

void knn::find_knearest(data * query_point)
{
  
}
void knn::set_training_data(std::vector<data * > * vect)
{
  train_data = vect;
}

void knn::set_test_data(std::vector<data *> * vect)
{
  test_data = vect;
}
void knn::set_validation_data(std::vector<data *> * vect)
{
  validation_data = vect;
}

void knn::set_k(int val)
{
  k = val;
}

int knn::predict();
double knn::calculate_distance(data * querypoint, data * input)
{
  double distance = 0.0;
  if (querypoint -> get_feature_vector_size() != input -> get_feature_vector_size())
  {
    printf("The feature vector size does not match.\n");
    exit(1);
  }

  #ifdef EUCLID
    for (unsigned i; i < querypoint -> get_feature_vector_size())
    {
      distance = pow(querypoint -> get_feature_vector() -> at(i) - input -> get_feature_vector() -> at(i), 2);
    }
    distance = sqrt(distance);

  #elif defined MANHATTAN
    //Here comes the implementation
  #endif
  return distance;
}

double knn::validate_performance();
double knn::test_performance();