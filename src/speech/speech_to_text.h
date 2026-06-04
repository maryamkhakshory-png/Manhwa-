#ifndef SPEECH_TO_TEXT_H
#define SPEECH_TO_TEXT_H

#include <string>
#include <vector>
#include <ffmpeg/avcodec.h>
#include <ffmpeg/avformat.h>

using namespace std;

class SpeechToText {
private:
    // Whisper model weights
    MatrixXd encoderWeights;
    MatrixXd decoderWeights;
    int sampleRate = 16000;
    
public:
    SpeechToText() {
        // Load pre-trained model
        loadModel("models/whisper_tiny.bin");
    }
    
    string transcribe(const string& audioFile) {
        // Extract audio features
        vector<float> features = extractFeatures(audioFile);
        
        // Run through encoder
        MatrixXd encoded = encode(features);
        
        // Decode to text
        string text = decode(encoded);
        
        return text;
    }
    
    void recordAndTranscribe(int duration = 5) {
        // Record from microphone
        system("termux-microphone-record -f temp.wav -d 5");
        
        // Transcribe
        string text = transcribe("temp.wav");
        
        cout << "Transcribed: " << text << endl;
    }
    
private:
    void loadModel(const string& path) {
        ifstream file(path, ios::binary);
        // Load weights...
    }
    
    vector<float> extractFeatures(const string& audioFile) {
        AVFormatContext* formatCtx = nullptr;
        avformat_open_input(&formatCtx, audioFile.c_str(), nullptr, nullptr);
        
        vector<float> features;
        // Extract MFCC features...
        
        avformat_close_input(&formatCtx);
        return features;
    }
    
    MatrixXd encode(const vector<float>& features) {
        // Run through transformer encoder
        int seqLen = features.size() / 80; // 80 mel bins
        MatrixXd input(seqLen, 80);
        
        // Copy features to matrix
        for(int i = 0; i < seqLen; i++) {
            for(int j = 0; j < 80; j++) {
                input(i, j) = features[i * 80 + j];
            }
        }
        
        return input * encoderWeights;
    }
    
    string decode(const MatrixXd& encoded) {
        // Decode using transformer decoder
        string result;
        // ... decoding logic
        return result;
    }
};

#endif
