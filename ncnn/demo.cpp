#include "net.h"
#include "opencv2/opencv.hpp"
#include "mrutil.h"

#define USE_INT8 0
#if _WIN32
    #pragma comment(lib,"ncnn.lib")
#endif

const char* class_names[] = {"face", "mask"};

std::string modelname = "mask";
#if USE_INT8
modelname = modelname + "-int8";
#endif

int input_size = 160;
const float mean_vals[3] = {127.5f, 127.5f, 127.5f};
const float norm_vals[3] = {1.0/127.5, 1.0/127.5, 1.0/127.5};
ncnn::Net net;

int detect(const cv::Mat& img, float threshold = 0.5){
    int img_h = img.size().height;
    int img_w = img.size().width;
    ncnn::Mat in = ncnn::Mat::from_pixels_resize(img.data, ncnn::Mat::PIXEL_BGR, img.cols, img.rows, input_size, input_size);
    in.substract_mean_normalize(mean_vals, norm_vals);
    ncnn::Mat out;
    ncnn::Extractor ex = net.create_extractor();
    ex.set_light_mode(true);
    ex.set_num_threads(4);
	cv::TickMeter tm;
	tm.start();
#if USE_PARAM_BIN
    ex.input(mobilenet_ssd_voc_ncnn_param_id::BLOB_data, in);
    ex.extract(mobilenet_ssd_voc_ncnn_param_id::BLOB_detection_out, out);    
#else
    ex.input("data", in);
    ex.extract("detection_out", out);
#endif
	tm.stop();
	std::cout << tm.getTimeMilli() << "ms" << std::endl;
    for (int i = 0; i < out.h; i++){
        const float *values = out.row(i);
        std::string label = class_names[int(values[0])-1] + double2string(values[1]);
        int x1 = values[2] * img_w;
        int y1 = values[3] * img_h;
        int x2 = values[4] * img_w;
        int y2 = values[5] * img_h;
        cv::rectangle(img, {x1,y1},{x2,y2},{255,0,0});
        cv::putText(img, label, {x1,y1},1,1,{0,0,255});
    }
    cv::imshow("img",img);
    cv::waitKey(1);
    return 0;
}

int test_image(std::string imagepath, int testnum = 1){
    cv::Mat img = cv::imread(imagepath);
    for (int i = 0; i < testnum; i++){
        detect(img);
    }
    cv::waitKey();
    return 0;
}

int test_camera() {
    cv::VideoCapture cap(0);
    cv::Mat img;
    while (true){
        cap >> img;
        if (!img.data) {
            break;
        }
        detect(img);
    }
    return 0;
}

int main(int argc, char*argv[]){
	#if NCNN_VULKAN
		ncnn::create_gpu_instance();
	#endif
	net.load_param((modelname + ".param").c_str());
	net.load_model((modelname + ".bin").c_str());
	std::string imgpath = "images/test.jpg";
	if (argc > 1) {
		imgpath = argv[1];
	}
    test_image(imgpath);
    //test_camera();
    return 0;
}