#include "mrdir.h"
#include "mropencv.h"
#include "SSDDetector.h"
using namespace ssd;

int testimage(std::string imgpath= "../images/000001.jpg"){
    cv::Mat img = cv::imread(imgpath);
    auto faces = Detect(img);
    auto show = drawDetection(img, faces);
    cv::imshow("img",show);
    cv::waitKey();
    return 0;
}

int testdir(const std::string imgdir="images"){
    auto files = getAllFilesinDir(imgdir,"*.jpg|*.png");
    for (int i = 0; i < files.size(); i++)
    {
        std::string filepath = imgdir + "/" + files[i];
        cv::Mat frame = cv::imread(filepath);
        cv::imshow("img", frame);
        cv::waitKey();
    }
    return 0;
}

int testcamera(int cameraindex = 0){
    cv::VideoCapture capture(cameraindex);
    cv::Mat frame;
    while (true){
        capture >> frame;
        if(!frame.data)
            break;
        auto faces = Detect(frame);
        auto show = drawDetection(frame, faces);
        imshow("SSDFace", show);
        waitKey(1);
    }
    return 0;
}

int main(int argc, char** argv){
    std::string modeldir="../";
    SSDConfig config;
    config.proto =  modeldir+"Face/MobileNetSSD_deploy.prototxt";
    config.model = modeldir+"Face/MobileNetSSD_face.caffemodel";
    init(config);
    // testimage();
    // testdir();
    testcamera();
    release();
}