#pragma once
#include "SSDFaceConfig.h"
#include "opencv2/opencv.hpp"
#include <opencv2/dnn.hpp>
#include <opencv2/core/utils/trace.hpp>
#include "mrutil.h"
using namespace cv::dnn;

class FaceBoxesDetector
{
public:
    static FaceBoxesDetector*getInstance()
    {
        static FaceBoxesDetector instance;
        return &instance;
    }
    int Detect(cv::Mat &img,float confidenceThreshold = 0.1);
    bool Init(const cv::String modelTxt,const cv::String modelBin);
private:
    cv::String _modelTxt = modelsdir + "/faceboxes_deploy.prototxt";
    cv::String _modelBin = modelsdir + "/FaceBoxes_1024x1024.caffemodel";
    cv::dnn::Net net;
    bool _bInited = false;
    FaceBoxesDetector(){
        Init(_modelTxt,_modelBin);
    }
    ~FaceBoxesDetector() {}
    cv::Mat getMean(const size_t& imageHeight, const size_t& imageWidth);
    cv::Mat preprocess(const cv::Mat& frame, const size_t width = 1024, const size_t height = 1024);
};

bool FaceBoxesDetector::Init(const cv::String modelTxt, const cv::String modelBin)
{
    _modelTxt = modelTxt;
    _modelBin = modelBin;
    try {
        net = cv::dnn::readNetFromCaffe(_modelTxt, _modelBin);
        _bInited = true;
    }
    catch (cv::Exception &e)
    {
        std::cout << e.what() << std::endl;
    }
    return _bInited;
}

int FaceBoxesDetector::Detect(cv::Mat &frame, float confidenceThreshold)
{
    if (!_bInited){
        Init(_modelTxt, _modelBin);
    }
    if (!_bInited){
        return -1;
    }
    cv::Mat preprocessedFrame = preprocess(frame);
    cv::Mat inputBlob = blobFromImage(preprocessedFrame);
    net.setInput(inputBlob, "data");
    cv::TickMeter tm;
    tm.start();
    cv::Mat detection = net.forward("detection_out");
    tm.stop();
    std::cout << tm.getTimeMilli() << "ms" << std::endl;
    cv::Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());
    for (int i = 0; i < detectionMat.rows; i++)
    {
        float confidence = detectionMat.at<float>(i, 2);
        if (confidence > confidenceThreshold)
        {
            size_t objectClass = (size_t)(detectionMat.at<float>(i, 1));
            float xLeftBottom = detectionMat.at<float>(i, 3) * frame.cols;
            float yLeftBottom = detectionMat.at<float>(i, 4) * frame.rows;
            float xRightTop = detectionMat.at<float>(i, 5) * frame.cols;
            float yRightTop = detectionMat.at<float>(i, 6) * frame.rows;
            std::cout << "Class: " << objectClass << std::endl;
            std::cout << "Confidence: " << confidence << std::endl;
            std::cout << " " << xLeftBottom
                << " " << yLeftBottom
                << " " << xRightTop
                << " " << yRightTop << std::endl;
            cv::Rect object((int)xLeftBottom, (int)yLeftBottom,
                (int)(xRightTop - xLeftBottom),
                (int)(yRightTop - yLeftBottom));
            rectangle(frame, object, Scalar(0, 255, 0));
            cv::putText(frame, "face:" + double2string(confidence), cv::Point(xLeftBottom, yLeftBottom), 1, 1, CV_RGB(255, 0, 0));
        }
    }

    return 0;
}

cv::Mat FaceBoxesDetector::getMean(const size_t& imageHeight, const size_t& imageWidth)
{
    cv::Mat mean;
    const int meanValues[3] = { 104, 117, 123 };
    std::vector<cv::Mat> meanChannels;
    for (int i = 0; i < 3; i++)
    {
        cv::Mat channel((int)imageHeight, (int)imageWidth, CV_32F, cv::Scalar(meanValues[i]));
        meanChannels.push_back(channel);
    }
    cv::merge(meanChannels, mean);
    return mean;
}

cv::Mat FaceBoxesDetector::preprocess(const cv::Mat& frame, const size_t width, const size_t height)
{
    cv::Mat preprocessed;
    frame.convertTo(preprocessed, CV_32FC3);
    cv::resize(preprocessed, preprocessed, cv::Size(width, height));
    cv::Mat mean = getMean(width, height);
    cv::subtract(preprocessed, mean, preprocessed);
    return preprocessed;
}