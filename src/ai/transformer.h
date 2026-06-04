#ifndef TRANSFORMER_H
#define TRANSFORMER_H

#include <vector>
#include <string>
#include <map>
#include <cmath>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

class Transformer {
private:
    int embedDim;
    int numHeads;
    int numLayers;
    int vocabSize;
    int maxSeqLen;
    
    // Embeddings
    MatrixXd tokenEmbedding;
    MatrixXd positionEmbedding;
    
    // Attention weights
    vector<MatrixXd> queryWeights;
    vector<MatrixXd> keyWeights;
    vector<MatrixXd> valueWeights;
    vector<MatrixXd> outputWeights;
    
    // Layer normalization
    double epsilon = 1e-6;
    
public:
    Transformer(int embed_dim, int heads, int layers, int vocab, int max_len)
        : embedDim(embed_dim), numHeads(heads), numLayers(layers),
          vocabSize(vocab), maxSeqLen(max_len) {
        
        random_device rd;
        mt19937 gen(rd());
        normal_distribution<> dist(0, 1);
        
        // Initialize embeddings
        tokenEmbedding = MatrixXd::Random(vocabSize, embedDim) * 0.02;
        positionEmbedding = MatrixXd::Random(maxSeqLen, embedDim) * 0.02;
        
        // Initialize attention weights
        int headDim = embedDim / heads;
        for(int i = 0; i < numLayers; i++) {
            queryWeights.push_back(MatrixXd::Random(embedDim, embedDim) * 0.02);
            keyWeights.push_back(MatrixXd::Random(embedDim, embedDim) * 0.02);
            valueWeights.push_back(MatrixXd::Random(embedDim, embedDim) * 0.02);
            outputWeights.push_back(MatrixXd::Random(embedDim, embedDim) * 0.02);
        }
    }
    
    MatrixXd forward(const MatrixXd& inputTokens) {
        int seqLen = inputTokens.rows();
        
        // Token + Position embeddings
        MatrixXd embedded = MatrixXd::Zero(seqLen, embedDim);
        for(int i = 0; i < seqLen; i++) {
            embedded.row(i) = tokenEmbedding.row(inputTokens(i, 0)) + 
                              positionEmbedding.row(i);
        }
        
        MatrixXd current = embedded;
        
        // Transformer layers
        for(int l = 0; l < numLayers; l++) {
            // Multi-head attention
            MatrixXd attentionOutput = multiHeadAttention(current, l);
            
            // Add & Norm
            current = layerNorm(current + attentionOutput);
            
            // Feed forward
            MatrixXd ffn = feedForward(current);
            
            // Add & Norm
            current = layerNorm(current + ffn);
        }
        
        return current;
    }
    
    string generate(const vector<int>& inputIds, int maxTokens, 
                   map<int, string>& idToToken) {
        MatrixXd input(inputIds.size(), 1);
        for(size_t i = 0; i < inputIds.size(); i++) {
            input(i, 0) = inputIds[i];
        }
        
        string result;
        for(int i = 0; i < maxTokens; i++) {
            MatrixXd output = forward(input);
            
            // Get last token prediction
            VectorXd lastToken = output.row(output.rows() - 1);
            
            // Sample from distribution
            int nextToken = sampleToken(lastToken);
            
            if(nextToken == 0) break; // End token
            
            result += idToToken[nextToken] + " ";
            
            // Add to input
            MatrixXd newInput(input.rows() + 1, 1);
            newInput << input, MatrixXd::Constant(1, 1, nextToken);
            input = newInput;
        }
        
        return result;
    }
    
private:
    MatrixXd multiHeadAttention(const MatrixXd& x, int layerIdx) {
        int seqLen = x.rows();
        int headDim = embedDim / numHeads;
        
        // Linear projections
        MatrixXd Q = x * queryWeights[layerIdx];
        MatrixXd K = x * keyWeights[layerIdx];
        MatrixXd V = x * valueWeights[layerIdx];
        
        // Split into heads
        MatrixXd attentionOutput = MatrixXd::Zero(seqLen, embedDim);
        
        for(int h = 0; h < numHeads; h++) {
            // Get head slices
            MatrixXd Qh = Q.block(0, h * headDim, seqLen, headDim);
            MatrixXd Kh = K.block(0, h * headDim, seqLen, headDim);
            MatrixXd Vh = V.block(0, h * headDim, seqLen, headDim);
            
            // Scaled dot-product attention
            MatrixXd scores = (Qh * Kh.transpose()) / sqrt(headDim);
            
            // Softmax
            for(int i = 0; i < scores.rows(); i++) {
                double maxVal = scores.row(i).maxCoeff();
                scores.row(i) = (scores.row(i).array() - maxVal).exp();
                scores.row(i) /= scores.row(i).sum();
            }
            
            // Apply attention to values
            attentionOutput.block(0, h * headDim, seqLen, headDim) = scores * Vh;
        }
        
        return attentionOutput * outputWeights[layerIdx];
    }
    
    MatrixXd feedForward(const MatrixXd& x) {
        // Simple feed-forward with ReLU
        int hiddenDim = embedDim * 4;
        MatrixXd hidden = x * MatrixXd::Random(embedDim, hiddenDim) * 0.02;
        
        // ReLU activation
        for(int i = 0; i < hidden.rows(); i++) {
            for(int j = 0; j < hidden.cols(); j++) {
                if(hidden(i, j) < 0) hidden(i, j) = 0;
            }
        }
        
        return hidden * MatrixXd::Random(hiddenDim, embedDim) * 0.02;
    }
    
    MatrixXd layerNorm(const MatrixXd& x) {
        MatrixXd normalized = x;
        for(int i = 0; i < x.rows(); i++) {
            double mean = x.row(i).mean();
            double var = (x.row(i).array() - mean).square().mean();
            normalized.row(i) = (x.row(i).array() - mean) / sqrt(var + epsilon);
        }
        return normalized;
    }
    
    int sampleToken(const VectorXd& logits) {
        // Temperature sampling
        double temperature = 0.8;
        VectorXd probs = (logits.array() / temperature).exp();
        probs /= probs.sum();
        
        // Random sampling
        random_device rd;
        mt19937 gen(rd());
        discrete_distribution<> dist(probs.data(), probs.data() + probs.size());
        
        return dist(gen);
    }
};

#endif
