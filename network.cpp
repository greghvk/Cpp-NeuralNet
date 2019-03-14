//
//  network.cpp
//  neuralNet
//
//  Created by Grzegorz Huk on 13/03/2019.
//  Copyright © 2019 Grzegorz Huk. All rights reserved.
//

#include "network.hpp"

//------------------------ Neuron ------------------------

double Neuron::eta = 0.2;
double Neuron::alpha = 0.2;

Neuron::Neuron(unsigned int outputs_num, int index_in)
{
    index = index_in;
    for(int i=0; i<outputs_num; i++)
    {
        out_connections.push_back(link());
        out_connections.back().weight = RandomWeight();
    }
};

void Neuron::ForwardProp(const Layer& prev_layer)
{
    double sum = 0.0;
    for(double input = 0; input < prev_layer.size(); input++)
    {
        sum += prev_layer[input].GetOutputVal() * prev_layer[input].out_connections[index].weight;
    };
    output_val = sigmoid(sum);
};

double Neuron::RandomWeight() {return random() / double(RAND_MAX);};
void Neuron::CalculateOutGradient(double expected_value)
{
    double delta = expected_value - output_val;
    gradient = delta * sigmoidDerivative(output_val);
};

void Neuron::CalculateHiddenGradient(const Layer &next_layer)
{
    double err_sum = CalcErrSum(next_layer);
    gradient = err_sum * sigmoidDerivative(output_val);
};

double Neuron::CalcErrSum(const Layer &next_layer)
{
    double sum = 0.0;
    
    for(int i = 0; i < next_layer.size() - 1; i++)
    {
        sum += out_connections[i].weight * next_layer[i].gradient;
    };
    
    return sum;
};

void Neuron::UpdateInWeights(Layer &prev_layer)
{
    for(int i = 0; i < prev_layer.size(); i++)
    {
        Neuron &neuron = prev_layer[i];
        double old_weight_change = neuron.out_connections[index].weight_change;
        
        double new_weight_change = (eta * neuron.GetOutputVal() * gradient) + (alpha * old_weight_change);
        
        neuron.out_connections[index].weight += new_weight_change;
        neuron.out_connections[index].weight_change = new_weight_change;
    }
    
};
//------------------------ Network ------------------------

Network::Network(const std::vector<unsigned int> &neurons_per_layer)
{
    unsigned long layers_num = neurons_per_layer.size();
    
    for(int current_layer = 0; current_layer < layers_num; current_layer++)
    {
        layers.push_back(Layer());
        //Check if current layer is the last layer - if so, add no connections
        //to neurons in this layer
        unsigned int next_layer_num = current_layer == neurons_per_layer.size()-1 ? 0 : neurons_per_layer[current_layer + 1];
        
        for(int current_neuron = 0; current_neuron <= neurons_per_layer[current_layer]; current_neuron++)
        {
            layers[current_layer].push_back(Neuron(next_layer_num, current_neuron));
            std::cout << "created neuron " << current_neuron << " of layer " << current_layer << std::endl;
        }
        layers.back().back().SetOutput(1.0);
        
    }
}

void Network::ForwardProp(std::vector<double> &input_vec)
{
    assert(input_vec.size()==layers[0].size()-1);
    
    //set values in the input layer of neurons
    for(int neuron_num=0; neuron_num<input_vec.size(); neuron_num++)
    {
        layers[0][neuron_num].SetOutput(input_vec[neuron_num]);
    };
    
    //
    for(int layer_num = 1; layer_num < layers.size(); layer_num++)
    {
        Layer &prev_layer = layers[layer_num-1];
        for(int neuron_num = 0; neuron_num < layers[layer_num].size() - 1; neuron_num++)
        {
            layers[layer_num][neuron_num].ForwardProp(prev_layer);
        };
    };
};

void Network::BackProp(std::vector<double> &expected_output)
{
    //Get network error - here root mean square
    Layer& output_layer = layers.back();
    net_error = 0.0;
    double delta;
    
    int out_size = output_layer.size() - 1;
    
    for(int i = 0; i < out_size; i++) //Errors not including bias neuron
    {
        delta = (expected_output[i] - output_layer[i].GetOutputVal());
        net_error += delta * delta;
    }
    
    net_error = sqrt(net_error / out_size);
    
    recent_average_error = net_error;//(recent_average_error * average_error_smoothing_factor + net_error) / (average_error_smoothing_factor + 1);
    
    //calculate gradients on output
    for (int i = 0; i < out_size; i++)
    {
        output_layer[i].CalculateOutGradient(expected_output[i]);
    }
    //gradient on hidden layers
    for(int i = layers.size() - 2; i > 0; i--)
    {
        Layer &current_hidden_layer = layers[i];
        Layer &next_layer = layers[i+1];
        
        for(int j = 0; j < current_hidden_layer.size(); j++)
        {
            current_hidden_layer[j].CalculateHiddenGradient(next_layer);
        }
    }
    //update the weights
    for(int i = layers.size()-1; i > 0; i--)
    {
        Layer &current_layer = layers[i];
        Layer &prev_layer = layers[i-1];
        
        for(int j = 0; j < current_layer.size() - 1; j++)
        {
            current_layer[j].UpdateInWeights(prev_layer);
        }
    }
    
    
};

void Network::GetOutput(std::vector<double> &results_vector)
{
    results_vector.clear();
    Layer &last_layer = layers.back();
    
    for(int i = 0; i < last_layer.size() - 1; i++)
    {
        results_vector.push_back(last_layer[i].GetOutputVal());
    }
    
};

