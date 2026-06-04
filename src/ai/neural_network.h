#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <vector>
#include <cmath>
#include <random>
#include <memory>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

class NeuralNetwork {
private:
    vector<MatrixXd> weights;
    vector<VectorXd> biases;
    vector<int> layers;
    double learningRate;
    
public:
    NeuralNetwork(vector<int> architecture, double lr = 0.01) 
        : layers(architecture), learningRate(lr) {
        random_device rd;
        mt19937 gen(rd());
        normal_distribution<> dist(0, 1);
        
        for(size_t i = 1; i < layers.size(); i++) {
            MatrixXd w(layers[i], layers[i-1]);
            for(int r = 0; r < w.rows(); r++) {
                for(int c = 0; c < w.cols(); c++) {
                    w(r, c) = dist(gen) * sqrt(2.0 / layers[i-1]);
                }
            }
            weights.push_back(w);
            biases.push_back(VectorXd::Zero(layers[i]));
        }
    }
    
    VectorXd forward(const VectorXd& input) {
        VectorXd current = input;
        for(size_t i = 0; i < weights.size(); i++) {
            VectorXd z = weights[i] * current + biases[i];
            current = activation(z);
        }
        return current;
    }
    
    void train(const vector<pair<VectorXd, VectorXd>>& data, int epochs) {
        for(int epoch = 0; epoch < epochs; epoch++) {
            double totalLoss = 0;
            for(const auto& [input, target] : data) {
                // Forward pass
                vector<VectorXd> activations = {input};
                vector<VectorXd> zs;
                VectorXd current = input;
                
                for(size_t i = 0; i < weights.size(); i++) {
                    VectorXd z = weights[i] * current + biases[i];
                    zs.push_back(z);
                    current = activation(z);
                    activations.push_back(current);
                }
                
                // Backward pass
                VectorXd delta = (current - target).cwiseProduct(activationDerivative(zs.back()));
                
                for(int i = weights.size() - 1; i >= 0; i--) {
                    VectorXd biasGrad = delta;
                    MatrixXd weightGrad = delta * activations[i].transpose();
                    
                    weights[i] -= learningRate * weightGrad;
                    biases[i] -= learningRate * biasGrad;
                    
                    if(i > 0) {
                        delta = (weights[i].transpose() * delta)
                               .cwiseProduct(activationDerivative(zs[i-1]));
                    }
                }
                
                totalLoss += (current - target).squaredNorm();
            }
            if(epoch % 100 == 0) {
                cout << "Epoch " << epoch << " Loss: " << totalLoss / data.size() << endl;
            }
        }
    }
    
private:
    VectorXd activation(const VectorXd& x) {
        return 1.0 / (1.0 + (-x.array()).exp());
    }
    
    VectorXd activationDerivative(const VectorXd& x) {
        VectorXd sig = activation(x);
        return sig.array() * (1 - sig.array());
    }
};

#endif
