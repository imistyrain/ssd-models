#ifndef FACE_DETECOR_H
#define FACE_DETECOR_H
#include "opencv2/opencv.hpp"

namespace ssd{
    struct DetectionResult{
        size_t classid;
        double confidence;
        cv::Rect r;
    };
    enum PREPROCESS_TYPE{
        MOBILENETSSD=0,
        SSD=1
    };

    struct SSDConfig{
        std::string proto;
        std::string model;
        int width = 0;
        int height = 0;
        PREPROCESS_TYPE type = MOBILENETSSD;
    };

    int init(const SSDConfig &config);

    std::vector<DetectionResult> Detect(const cv::Mat &img);

    cv::Mat drawDetection(const cv::Mat &img, std::vector<DetectionResult>&results);

    int release();
}

#endif