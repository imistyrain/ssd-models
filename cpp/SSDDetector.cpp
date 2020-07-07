#include "config.h"

#if USE_CAFFE
    #include <caffe/caffe.hpp>
    using namespace caffe;
    caffe::Net<float> *g_net;
#endif

#include "SSDDetector.h"
#include "mrutil.h"

namespace ssd{
int g_inputH = 300;
int g_inputW = 300;
PREPROCESS_TYPE g_preprocess_type = MOBILENETSSD;

#if USE_DNN
    #include <opencv2/dnn.hpp>
    using namespace cv::dnn;
    cv::dnn::Net g_net;
#endif

    int init(const SSDConfig &config){
        if (config.width > 0){
            g_inputW = config.width;
        }
        if (config.height > 0){
            g_inputH = config.height;
        }
        g_preprocess_type = config.type;
        #if USE_CAFFE
            g_net = new caffe::Net<float>(config.proto, caffe::TEST);
            g_net->CopyTrainedLayersFrom(config.model);
            if (g_net)
            {
                auto inputblob = g_net->input_blobs()[0];
                g_inputH = inputblob->height();
                g_inputW = inputblob->width();
            }
        #endif

        #if USE_DNN
            g_net = cv::dnn::readNetFromCaffe(config.proto,config.model);
        #endif

        return 0;
    }

    cv::Mat preprocess(const cv::Mat& img){
        cv::Mat preprocessed;
        img.convertTo(preprocessed, CV_32F);
        cv::resize(preprocessed, preprocessed, cv::Size(g_inputW, g_inputH));
        if (g_preprocess_type == SSD){
            cv::subtract(preprocessed, cv::Scalar(104,117,123), preprocessed);
        } else {
            preprocessed -= 127.5f;
            preprocessed *= 0.007843f;
        }
        return preprocessed;
    }
    std::vector<DetectionResult> Detect(const cv::Mat &img){
        std::vector<DetectionResult> detections;
        cv::Mat preprocessed = preprocess(img);
        #if USE_CAFFE
            Blob<float>* input_layer = g_net->input_blobs()[0];
            int width = input_layer->width();
            int height = input_layer->height();
            int channels = input_layer->channels();
            float* input_data = input_layer->mutable_cpu_data();
            for(int c = 0; c < channels;c++){
                for(int h=0;h<height;h++){
                    for(int w=0;w<width;w++){
                        input_data[c*width*height+h*width+w]=preprocessed.at<cv::Vec3f>(h,w)[c];
                    }
                }
            }
            g_net->Forward();
            Blob<float>* result_blob = g_net->output_blobs()[0];
            const float* result = result_blob->cpu_data();
            const int num_det = result_blob->height();
            for (int k = 0; k < num_det; ++k) {
                if (result[0] == -1) {
                    result += 7;
                    continue;
                }
                DetectionResult detection;
                detection.classid = result[1];
                detection.confidence = result[2];
                detection.r.x = result[3]*img.cols;
                detection.r.y = result[4]*img.rows;
                detection.r.width = (result[5]-result[3])*img.cols;
                detection.r.height = (result[6]-result[4])*img.rows;
                detections.push_back(detection);
                result += 7;
            }
        #endif
        #if USE_DNN
            cv::Mat inputBlob = blobFromImage(preprocessed);
            g_net.setInput(inputBlob, "data");
            cv::Mat detection = g_net.forward("detection_out");
            cv::Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());
            for (int i = 0; i < detectionMat.rows; i++)
            {
                float confidence = detectionMat.at<float>(i, 2);
                if (confidence > 0.1)
                {
                    DetectionResult dr;
                    size_t objectClass = (size_t)(detectionMat.at<float>(i, 1));
                    float xLeftBottom = detectionMat.at<float>(i, 3) * img.cols;
                    float yLeftBottom = detectionMat.at<float>(i, 4) * img.rows;
                    float xRightTop = detectionMat.at<float>(i, 5) * img.cols;
                    float yRightTop = detectionMat.at<float>(i, 6) * img.rows;
                    cv::Rect object((int)xLeftBottom, (int)yLeftBottom,
                        (int)(xRightTop - xLeftBottom),
                        (int)(yRightTop - yLeftBottom));
                        dr.classid = objectClass;
                    dr.confidence = confidence;
                    dr.r = object;
                    detections.push_back(dr);
                }
            }   
        #endif
        return detections;
    }

    cv::Mat drawDetection(const cv::Mat &img, std::vector<DetectionResult>&results)
    {
        cv::Mat show = img.clone();
        for (int i = 0; i < results.size(); i++)
        {
            auto dr=results[i];
            cv::rectangle(show, dr.r, cv::Scalar(0, 255, 0));
            std::string title = "Face:" + double2string(dr.confidence);
            cv::putText(show, title, cv::Point(dr.r.x,dr.r.y), 1, 1,CV_RGB(255, 0, 0));
        }
        return show;
    }

    int release()
    {
        #if USE_CAFFE
            delete g_net;
        #endif
        return true;
    }
}