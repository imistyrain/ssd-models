#include <fstream>
#include <iostream>
#include <cstdlib>
#include <opencv2/dnn.hpp>
#include <opencv2/dnn/shape_utils.hpp>
#include "mropencv.h"
#include "mrutil.h"
using namespace cv::dnn;
using namespace std;
const string modelsdir = "models";
String modelTxt = modelsdir+"/face_deploy.prototxt";
String modelBin = modelsdir + "/VGG_Face2017_SSD_300x300_iter_120000.caffemodel";
const size_t width = 300;
const size_t height = 300;
static Mat getMean(const size_t& imageHeight, const size_t& imageWidth)
{
    Mat mean;

    const int meanValues[3] = {104, 117, 123};
    vector<Mat> meanChannels;
    for(int i = 0; i < 3; i++)
    {
        Mat channel((int)imageHeight, (int)imageWidth, CV_32F, Scalar(meanValues[i]));
        meanChannels.push_back(channel);
    }
    cv::merge(meanChannels, mean);
    return mean;
}
static Mat preprocess(const Mat& frame)
{
    Mat preprocessed;
    frame.convertTo(preprocessed, CV_32F);
    resize(preprocessed, preprocessed, Size(width, height)); //SSD accepts 300x300 RGB-images

    Mat mean = getMean(width, height);
    cv::subtract(preprocessed, mean, preprocessed);

    return preprocessed;
}
class SSDFaceDetector
{
public:
	static SSDFaceDetector*getInstance()
	{
		static SSDFaceDetector instance;
		return &instance;
	}
	int Detect(cv::Mat &img);
private:
	dnn::Net net;
	SSDFaceDetector() 
	{
		net = dnn::readNetFromCaffe(modelTxt, modelBin);
	}
	~SSDFaceDetector() {}
};

int SSDFaceDetector::Detect(cv::Mat &frame)
{
	Mat preprocessedFrame = preprocess(frame);
	Mat inputBlob = blobFromImage(preprocessedFrame);
	net.setInput(inputBlob, "data");
	cv::TickMeter tm;
	tm.start();
	Mat detection = net.forward("detection_out");
	tm.stop();
	cout << tm.getTimeMilli() << "ms" << endl;
	Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());
	float confidenceThreshold = 0.1;
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
			Rect object((int)xLeftBottom, (int)yLeftBottom,
				(int)(xRightTop - xLeftBottom),
				(int)(yRightTop - yLeftBottom));
			rectangle(frame, object, Scalar(0, 255, 0));
			cv::putText(frame, "face:"+double2string(confidence), cv::Point(xLeftBottom, yLeftBottom), 1, 1, CV_RGB(255, 0, 0));
		}
	}

	return 0;
}

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
        imshow("SSDFace", frame);
        waitKey(1);
    }
    return 0;
}
int testimage(int argc, char** argv)
{
	string filepath= "000001.jpg";
	if (argc > 1)
		filepath = argv[1];
	cv::Mat frame = cv::imread(filepath, -1);
    if (frame.channels() == 4)
        cvtColor(frame, frame, COLOR_BGRA2BGR);
	SSDFaceDetector::getInstance()->Detect(frame);
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