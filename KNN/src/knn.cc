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
  // Initialize a dynamic vector to store pointers to the k nearest neighbors
  neighbors = new std::vector<data *>;

  // Initialize min with the maximum possible double value to ensure any real distance
  // will be smaller and get properly compared
  double min = std::numeric_limits<double>::max();

  // Keeps track of the previous minimum distance found to avoid selecting the same neighbor twice
  double previous_min = min;

  // Will store the index/position of the current nearest neighbor in the training data
  int index = 0;

  // Outer loop: runs k times, finding one neighbor per iteration
  for (int i = 0 ; i < k ; i++)
  {
    // First iteration (i == 0): Find the closest neighbor to the query point
    if(i == 0)
    {
      // Inner loop: iterate through all training data points
      for (int j = 0; j < train_data -> size(); j++)
      {
        // Calculate Euclidean distance between query_point and current training data point
        double distance = calculate_distance(query_point, train_data -> at(j));

        // Store the calculated distance in the training data point
        train_data -> at(i) -> set_distance(distance);

        // If this distance is smaller than the current minimum, update min and index
        if (distance < min)
        {
          min = distance;
          index = j;
        }
      }

      // Add the closest neighbor (at position index) to the neighbors vector
      neighbors -> push_back(train_data -> at(index));

      // Save the current minimum distance as the previous minimum
      // This will be used in the next iteration to avoid duplicates
      previous_min = min;

      // Reset min to maximum value for the next neighbor search
      min = std::numeric_limits<double>::max();
    }
    // Subsequent iterations (i > 0): Find the next k-1 nearest neighbors
    else
    {
      // Inner loop: iterate through all training data points again
      for (int j=0 ; j < train_data -> size(); j++)
      {
        // Calculate distance between query_point and current training data point
        double distance = calculate_distance(query_point, train_data ->at(j));

        // Only consider distances that are:
        // 1. Greater than previous_min (avoids selecting previously found neighbors)
        // 2. Smaller than current min (finds the next closest unselected neighbor)
        if (distance > previous_min && distance < min)
        {
          min = distance;
          index = j;
        }
      }

      // Add the next nearest neighbor (at position index) to the neighbors vector
      neighbors -> push_back(train_data -> at(index));

      // Update previous_min to the current minimum distance found
      previous_min = min;

      // Reset min to maximum value for the next neighbor search
      min = std::numeric_limits<double>::max();
    }
  }
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

int knn::predict()
{
  std::map<uint8_t, int> class_freq;

  // Count the occurrences of each label among the k nearest neighbors
  for (int i = 0; i < neighbors -> size(); i++)
  {
    if (class_freq.find(neighbors -> at(i) -> get_label()) == class_freq.end())
    {
      class_freq[neighbors -> at(i) -> get_label()] = 1;
    }
    else
    {
      class_freq[neighbors -> at(i) -> get_label()] += 1;
    }
  }

  int best = 0;
  int max = 0;

  for (auto kv: class_freq)
  {
    if (kv.second > max)
    {
      max = kv.second;
      best = kv.first;
    }
  }
  delete neighbors;
  return best;
}

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

double knn::validate_performance()
{
  double current_performance = 0;
  int count = 0;
  int data_index = 0;
  for(data * query_point : *validation_data)
  {
    find_knearest(query_point);
    int predicted_label = predict();
    if (predicted_label == query_point -> get_label())
    {
      count++;
    }
    data_index++;
  }
  current_performance = (double)count * 100.0 / (double)data_index;
  printf("Validation Accuracy: %.2f %%\n", current_performance);
  return current_performance;

}

double knn::test_performance();