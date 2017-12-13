#include "mropencv.h"
#include "SSDFaceDetector.h"
#include "FaceBoxes.h"
int testcamera(int cameraindex=0)
{
    cv::VideoCapture capture(cameraindex);
    cv::Mat frame;
    while (true)
    {
        capture >> frame;
        if(!frame.data)
            break;
        SSDFaceDetector::getInstance()->Detect(frame);
//        FaceBoxesDetector::getInstance()->Detect(frame);
        imshow("SSDFace", frame);
        waitKey(1);
    }
    return 0;
}
int testimage(int argc, char** argv)
{
    std::string filepath= "000001.jpg";
	if (argc > 1)
		filepath = argv[1];
	cv::Mat frame = cv::imread(filepath);
    cv::namedWindow("SSDFace", 0);
	SSDFaceDetector::getInstance()->Detect(frame);
//    FaceBoxesDetector::getInstance()->Detect(frame);
	imshow("SSDFace", frame);
	waitKey();
    return 0;
}

int main(int argc, char** argv)
{
    testimage(argc, argv);
//    testcamera();
    return 0;
}