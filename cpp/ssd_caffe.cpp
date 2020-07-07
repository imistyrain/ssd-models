#include <caffe/caffe.hpp>
#include "mrdir.h"
#include "mropencv.h"
#include <algorithm>
#include <iomanip>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>

using namespace caffe;  // NOLINT(build/namespaces)

class Detector {
public:
	Detector(const string& model_file,const string& weights_file);

	std::vector<vector<float> > Detect(const cv::Mat& img);

private:
	void WrapInputLayer(std::vector<cv::Mat>* input_channels);

	void Preprocess(const cv::Mat& img,
		std::vector<cv::Mat>* input_channels);

private:
	shared_ptr<Net<float> > net_;
	cv::Size input_geometry_;
	int num_channels_;
};

Detector::Detector(const string& model_file,
	const string& weights_file) {
#ifdef CPU_ONLY
	Caffe::set_mode(Caffe::CPU);
#else
	Caffe::set_mode(Caffe::GPU);
#endif

	/* Load the network. */
	net_.reset(new Net<float>(model_file, TEST));
	net_->CopyTrainedLayersFrom(weights_file);

	CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
	CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

	Blob<float>* input_layer = net_->input_blobs()[0];
	num_channels_ = input_layer->channels();
	CHECK(num_channels_ == 3 || num_channels_ == 1)
		<< "Input layer should have 1 or 3 channels.";
	input_geometry_ = cv::Size(input_layer->width(), input_layer->height());
}

std::vector<vector<float> > Detector::Detect(const cv::Mat& img) {
	Blob<float>* input_layer = net_->input_blobs()[0];
	input_layer->Reshape(1, num_channels_,
		input_geometry_.height, input_geometry_.width);
	/* Forward dimension change to all layers. */
	net_->Reshape();

	std::vector<cv::Mat> input_channels;
	WrapInputLayer(&input_channels);

	Preprocess(img, &input_channels);

	net_->Forward();

	/* Copy the output layer to a std::vector */
	Blob<float>* result_blob = net_->output_blobs()[0];
	const float* result = result_blob->cpu_data();
	const int num_det = result_blob->height();
	vector<vector<float> > detections;
	for (int k = 0; k < num_det; ++k) {
		if (result[0] == -1) {
			// Skip invalid detection.
			result += 7;
			continue;
		}
		vector<float> detection(result, result + 7);
		detections.push_back(detection);
		result += 7;
	}
	return detections;
}

/* Wrap the input layer of the network in separate cv::Mat objects
* (one per channel). This way we save one memcpy operation and we
* don't need to rely on cudaMemcpy2D. The last preprocessing
* operation will write the separate channels directly to the input
* layer. */
void Detector::WrapInputLayer(std::vector<cv::Mat>* input_channels) {
	Blob<float>* input_layer = net_->input_blobs()[0];

	int width = input_layer->width();
	int height = input_layer->height();
	float* input_data = input_layer->mutable_cpu_data();
	for (int i = 0; i < input_layer->channels(); ++i) {
		cv::Mat channel(height, width, CV_32FC1, input_data);
		input_channels->push_back(channel);
		input_data += width * height;
	}
}

void Detector::Preprocess(const cv::Mat& img,
	std::vector<cv::Mat>* input_channels) {
	/* Convert the input image to the input image format of the network. */
	cv::Mat sample;
	if (img.channels() == 3 && num_channels_ == 1)
		cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
	else if (img.channels() == 4 && num_channels_ == 1)
		cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
	else if (img.channels() == 4 && num_channels_ == 3)
		cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
	else if (img.channels() == 1 && num_channels_ == 3)
		cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
	else
		sample = img;

	cv::Mat sample_resized;
	if (sample.size() != input_geometry_)
		cv::resize(sample, sample_resized, input_geometry_);
	else
		sample_resized = sample;

	cv::Mat sample_float;
	if (num_channels_ == 3)
		sample_resized.convertTo(sample_float, CV_32FC3);
	else
		sample_resized.convertTo(sample_float, CV_32FC1);

	cv::Mat sample_normalized;
//	cv::subtract(sample_float, mean_, sample_normalized);
	sample_normalized = (sample_float - 127.5)*0.007843;
	/* This operation will write the separate BGR planes directly to the
	* input layer of the network because it is wrapped by the cv::Mat
	* objects in input_channels. */
	cv::split(sample_normalized, *input_channels);

	CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
		== net_->input_blobs()[0]->cpu_data())
		<< "Input channels are not wrapping the input layer of the network.";
}
DEFINE_double(confidence_threshold, 0.9,
	"Only store detections with score higher than the threshold.");
#include "fstream"
static void split(const std::string& s, const std::string delim, std::vector< std::string >* ret)
{
	size_t last = 0;
	size_t index = s.find_first_of(delim, last);
	while (index != std::string::npos)
	{
		ret->push_back(s.substr(last, index - last));
		last = index + 1;
		index = s.find_first_of(delim, last);
	}
	if (index - last > 0)
	{
		ret->push_back(s.substr(last, index - last));
	}
}
vector<string>getLabels(const string labeltxt)
{
	vector<string>labels;
	std::ifstream labelmapfile(labeltxt);
	string line;
	while (getline(labelmapfile, line))
	{
		vector<string>strs;
		split(line, ":", &strs);
		if (strs.size() == 2 && strs[0] == "  name")
		{
			string label = strs[1];
			label = label.substr(2, label.length() - 3);
			labels.push_back(label);
		}
	}
	return labels;
}

cv::Mat ShowDetectionResult(const cv::Mat &img, const std::vector<vector<float> > &detections, std::vector<std::string> &labels, double confidence_threshold)
{
	cv::Mat show=img.clone();
	for (int i = 0; i < detections.size(); ++i)
	{
		const vector<float>& d = detections[i];
		// Detection format: [image_id, label, score, xmin, ymin, xmax, ymax].
		CHECK_EQ(d.size(), 7);
		const float score = d[2];
		if (score >= confidence_threshold)
		{
			cv::rectangle(show, cv::Point(d[3] * img.cols, d[4] * img.rows), cv::Point(d[5] * img.cols, d[6] * img.rows), CV_RGB(255, 0, 0));
			cv::putText(show, labels[static_cast<int>(d[1])], cv::Point(d[3] * img.cols, d[4] * img.rows), 2, 1, CV_RGB(255, 0, 0));
		}
	}
	return show;
}

void testdir(Detector &detector, std::vector<std::string> &labels,double confidence_threshold,const string dir = "../images")
{
	auto files = getAllFilesinDir(dir);
	for (int i = 0; i < files.size(); i++)
	{
		std::string filepath = dir + "/" + files[i];
		cv::Mat img = cv::imread(filepath, -1);
		std::cout << files[i] << std::endl;
		CHECK(!img.empty()) << "Unable to decode image " << filepath;
		cv::TickMeter tm;
		tm.start();
		std::vector<vector<float> > detections = detector.Detect(img);
		tm.stop();
		std::cout << tm.getTimeMilli() << "ms" << std::endl;
		/* Print the detection results. */
		cv::Mat show = ShowDetectionResult(img, detections,labels,confidence_threshold);
		cv::imshow("img", show);
		cv::waitKey();
		}
	cv::waitKey();
}

void testcamera(Detector &detector, std::vector<std::string> &labels, double confidence_threshold, int cameraindex = 0)
{
	cv::VideoCapture cap(cameraindex);
	if (!cap.isOpened()) {
		LOG(FATAL) << "Failed to open camera ";
	}
	cv::Mat img;
	int frame_count = 0;
	while (true) {
		bool success = cap.read(img);
		if (!success) {
			LOG(INFO) << "Process " << frame_count << " frames";
			break;
		}
		CHECK(!img.empty()) << "Error when read frame";
		cv::TickMeter tm;
		tm.start();
		std::vector<vector<float> > detections = detector.Detect(img);
		tm.stop();
		std::cout << tm.getTimeMilli() << "ms"<<std::endl;
		cv::Mat show = ShowDetectionResult(img, detections, labels, confidence_threshold);
		++frame_count;
		cv::imshow("img", show);
		cv::waitKey(1);
	}
	if (cap.isOpened()) {
		cap.release();
	}
}

int main(int argc, char** argv) {
	::google::InitGoogleLogging(argv[0]);
	// Print output to stderr (while still logging)
	FLAGS_alsologtostderr = 1;

#ifndef GFLAGS_GFLAGS_H_
	namespace gflags = google;
#endif

	gflags::SetUsageMessage("Do detection using SSD mode.\n"
		"Usage:\n"
		"    ssd_detect [FLAGS] model_file weights_file list_file\n");
	gflags::ParseCommandLineFlags(&argc, &argv, true);
	const string& dataset = "Face";
	const string& labelfilepath = "../" + dataset + "/labelmap.prototxt";
	const string& model_file = "../"+dataset+"/MobileNetSSD_deploy_depth.prototxt";
	const string& weights_file = "../" + dataset + "/MobileNetSSD_face.caffemodel";
	const float confidence_threshold = FLAGS_confidence_threshold;
	// Initialize the network.
	Detector detector(model_file, weights_file);
	auto labels = getLabels(labelfilepath);
	testdir(detector, labels, confidence_threshold);
	//testcamera(detector, labels, confidence_threshold);
	return 0;
}