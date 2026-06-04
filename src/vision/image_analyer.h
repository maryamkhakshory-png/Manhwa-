#ifndef IMAGE_ANALYZER_H
#define IMAGE_ANALYZER_H

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <tesseract/baseapi.h>
#include <string>

using namespace cv;
using namespace std;

class ImageAnalyzer {
private:
    dnn::Net yoloNet;
    dnn::Net resNet;
    tesseract::TessBaseAPI tessApi;
    vector<string> classNames;
    
public:
    ImageAnalyzer() {
        // Load YOLO for object detection
        yoloNet = dnn::readNet("models/yolov4.weights", "models/yolov4.cfg");
        
        // Load ResNet for classification
        resNet = dnn::readNet("models/resnet50.weights", "models/resnet50.cfg");
        
        // Initialize Tesseract for OCR
        tessApi.Init(NULL, "fas+eng");
        
        // Load class names
        ifstream file("data/coco.names");
        string line;
        while(getline(file, line)) {
            classNames.push_back(line);
        }
    }
    
    vector<DetectedObject> detectObjects(const Mat& image) {
        Mat blob = dnn::blobFromImage(image, 1/255.0, Size(416, 416), Scalar(), true, false);
        yoloNet.setInput(blob);
        
        vector<Mat> outputs;
        yoloNet.forward(outputs, getOutputNames(yoloNet));
        
        vector<DetectedObject> objects;
        
        for(const auto& output : outputs) {
            for(int i = 0; i < output.rows; i++) {
                Mat scores = output.row(i).colRange(5, output.cols);
                Point classIdPoint;
                double confidence;
                minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
                
                if(confidence > 0.5) {
                    int centerX = output.at<float>(i, 0) * image.cols;
                    int centerY = output.at<float>(i, 1) * image.rows;
                    int width = output.at<float>(i, 2) * image.cols;
                    int height = output.at<float>(i, 3) * image.rows;
                    
                    objects.push_back({
                        classNames[classIdPoint.x],
                        confidence,
                        Rect(centerX - width/2, centerY - height/2, width, height)
                    });
                }
            }
        }
        
        return objects;
    }
    
    string classifyImage(const Mat& image) {
        Mat blob = dnn::blobFromImage(image, 1.0, Size(224, 224), Scalar(104, 117, 123));
        resNet.setInput(blob);
        Mat output = resNet.forward();
        
        Point classId;
        double confidence;
        minMaxLoc(output, 0, &confidence, 0, &classId);
        
        return classNames[classId.x];
    }
    
    string extractText(const Mat& image) {
        tessApi.SetImage(image.data, image.cols, image.rows, 3, image.step);
        char* text = tessApi.GetUTF8Text();
        string result(text);
        delete[] text;
        return result;
    }
    
    Mat generateImage(const string& description) {
        // Use GAN for image generation (simplified)
        Mat generatedImage(256, 256, CV_8UC3);
        // ... GAN generation logic
        return generatedImage;
    }
    
private:
    struct DetectedObject {
        string className;
        double confidence;
        Rect boundingBox;
    };
    
    vector<string> getOutputNames(const dnn::Net& net) {
        vector<string> names;
        vector<int> outLayers = net.getUnconnectedOutLayers();
        vector<string> layersNames = net.getLayerNames();
        
        names.resize(outLayers.size());
        for(size_t i = 0; i < outLayers.size(); i++) {
            names[i] = layersNames[outLayers[i] - 1];
        }
        return names;
    }
};

#endif
