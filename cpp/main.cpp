#include "mrdir.h"
#include "mropencv.h"
#include "SSDFaceDetector.h"
#include "FaceBoxes.h"
#include "VOCUtil/DataSetConfig.h"
#include "VOCUtil/AnnotationFile.h"
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
//	SSDFaceDetector::getInstance()->Detect(frame);
    FaceBoxesDetector::getInstance()->Detect(frame);
	imshow("SSDFace", frame);
	waitKey();
    return 0;
}

int testdir(const std::string imgdir="./")
{
    auto files = getAllFilesinDir(imgdir,"*.jpg");
    for (int i = 0; i < files.size(); i++)
    {
        std::string filepath = imgdir + "/" + files[i];
        cv::Mat frame = cv::imread(filepath);
        cv::namedWindow("SSDFace", 0);
        SSDFaceDetector::getInstance()->Detect(frame);
        imshow("SSDFace", frame);
        waitKey();
    }
    return 0;
}

int test_ibm(const std::string imgdir = "./",const std::string annodir="Annotations")
{
    const std::string databasedir = "./";
    DatasetConfig voc;
    voc.init(databasedir);
//     std::vector<std::string>names;
//     names.push_back("background");
//     names.push_back("face");
    auto files = getAllFilesinDir(imgdir, "*.jpg");
    AnnotationFile af;
    for (int i = 0; i < files.size(); i++)
    {
        std::string filepath = imgdir + "/" + files[i];
        cv::Mat img = cv::imread(filepath);
        cv::namedWindow("SSDFace", 0);
        auto results=SSDFaceDetector::getInstance()->Detect(img);
        drawDetectionResults(img, results,voc.classes);
        af.set_width(img.cols);
        af.set_height(img.rows);
        af.set_depth(img.channels());
        af.set_filename(files[i]);
        vector<Object>objects;
        for (int i = 0; i < results.size(); i++)
        {
            auto dr=results[i];
            Object object;
            object.difficult = 0;
            object.truncated = 0;
            object.name =voc.classes[dr.classid];
            object.xmin = dr.r.x;
            object.ymin = dr.r.y;
            object.xmax = dr.r.x+dr.r.width;
            object.ymax = dr.r.y+dr.r.height;
            objects.push_back(object);
        }
        af.set_objects(objects);
        std::string annoxmlpath = annodir + "/" + files[i];
        annoxmlpath = annoxmlpath.substr(0, annoxmlpath.length() - 3) + "xml";
        af.save_xml(annoxmlpath);
        imshow("SSDFace", img);
        waitKey(1);
    }
    cv::waitKey();
    return 0;
}

int main(int argc, char** argv)
{
    testimage(argc, argv);
//    testcamera();
//    testdir();
//    test_ibm();
    return 0;
}