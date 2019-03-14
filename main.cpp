#include <vector>
#include "network.hpp"

void printVec(std::vector<double> vector)
{
    for(int i=0; i<vector.size(); i++)
    {
        std::cout << "Vector value "<< i << " is " << vector[i] << std::endl;
    };
};

int main(int argc, const char * argv[]) {
    
    std::vector<unsigned int> first;
    first.push_back(2);
    first.push_back(3);
    first.push_back(1);
    
    Network net(first);
    
    
    std::vector<double> testVec;
    std::vector<double> results;
    testVec.push_back(1);
    testVec.push_back(0);
    
    std::vector<double> input00;
    std::vector<double> input01;
    std::vector<double> input10;
    std::vector<double> input11;
    
    input00.push_back(0);
    input00.push_back(0);
    
    input01.push_back(0);
    input01.push_back(1);
    
    input10.push_back(1);
    input10.push_back(0);
    
    input11.push_back(1);
    input11.push_back(1);
    
    std::vector<double> resultVec1;
    std::vector<double> resultVec0;
    resultVec1.push_back(1);
    resultVec0.push_back(0);
    
    for(int i = 0; i < 2000; i++)
    {
        net.ForwardProp(input00);
        net.GetOutput(results);
        printVec(results);
        net.BackProp(resultVec0);
        
        net.ForwardProp(input01);
        net.GetOutput(results);
        printVec(results);
        net.BackProp(resultVec0);
        
        net.ForwardProp(input10);
        net.GetOutput(results);
        printVec(results);
        net.BackProp(resultVec0);
        
        net.ForwardProp(input11);
        net.GetOutput(results);
        printVec(results);
        net.BackProp(resultVec1);
    }
    net.GetOutput(results);
    printVec(results);
    
    
    
    
    
    std::cout <<"end";
    
}
