//
//  network.hpp
//  neuralNet
//
//  Created by Grzegorz Huk on 13/03/2019.
//  Copyright © 2019 Grzegorz Huk. All rights reserved.
//

#ifndef network_hpp
#define network_hpp

#include <stdio.h>
#include <vector>
#include <cmath>
#include <iostream>

struct link
{
    double weight;
    double weight_change;
};

class Neuron;

typedef std::vector<Neuron> Layer;

class Neuron
{
public:
    Neuron(unsigned int outputs_num, int index);
    void SetOutput(double output) {output_val = output;};
    double GetOutputVal() const {return output_val;};
    void ForwardProp(const Layer& prev_layer);
    void CalculateOutGradient(double expected_value);
    void CalculateHiddenGradient(const Layer &next_layer);
    void UpdateInWeights(Layer &prev_layer);
    double CalcErrSum(const Layer &next_layer);
    
private:
    
    double output_val;
    double gradient;
    unsigned int index;
    std::vector<link> out_connections;
    static double eta; //learning rate
    static double alpha; //momentum
    
    double RandomWeight();
    
};


class Network
{
public:
    Network(const std::vector<unsigned int> &neurons_per_layer);
    void ForwardProp(std::vector<double> &input_vec);
    void BackProp(std::vector<double> &expected_output);
    void GetOutput(std::vector<double> &results_vector);
    double GetRecentError() {return recent_average_error;};
    
    
private:
    std::vector<Layer> layers;//vector of layers which are vectors of neuron by layers[i][j]
    double net_error;
    double recent_average_error;
    double average_error_smoothing_factor;
};

double sigmoid(double value) {return 1 / (1 + exp(-value));}; //fast sigmoid function
double sigmoidDerivative(double value) {return sigmoid(value) * (1 - sigmoid(value));};

#endif /* network_hpp */
