#pragma once
#include "string"
#include "mrutil.h"
#include "opencv2/opencv.hpp"

const std::string modelsdir = "models";

class DetectionResult
{
public:
    size_t classid;
    double confidence;
    cv::Rect r;
};

static void drawDetectionResults(cv::Mat &img, std::vector<DetectionResult>&results,std::vector<std::string>names)
{
    for (int i = 0; i < results.size(); i++)
    {
        auto dr=results[i];
        cv::rectangle(img, dr.r, Scalar(0, 255, 0));
        std::string title = names[dr.classid] + ":" + double2string(dr.confidence);
        cv::putText(img, title,cv::Point(dr.r.x,dr.r.y), 1, 1,CV_RGB(255, 0, 0));
    }
}