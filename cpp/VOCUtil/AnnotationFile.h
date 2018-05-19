#pragma once
#include "vector"
#include "string"
#include "map"
#include "opencv2/opencv.hpp"
using namespace std;
#define OUTPUT_OBJECT_POSSIBILITY 0
class Object
{
public:
	string name;
	string pose;
	bool truncated=0;
	bool difficult=0;
#if OUTPUT_OBJECT_POSSIBILITY
	double possibility=1;
#endif
	int xmin, ymin, xmax, ymax;
};
class AnnotationFile
{
public:
	string folder;
	string filename;
	int width, height, depth;
	bool segmented;
	string database;
	string annotation;
	string image;
	vector<Object>objects;
	static map<string, int>labelmap;
	static void set_labelmaps(const vector<string> classes);
	void set_filename(const string filename){ this->filename = filename; };
	void set_width(const int width){this->width = width;};
	void set_height(const int height){this->height = height;};
	void set_depth(const int depth){ this->depth = depth; };
	void set_objects(const vector<Object>&objects){ this->objects = objects; };
	bool load_xml(const string annotationfilepath);
	void save_xml(const string xmlannotationfilepath);
	bool load_txt(const string txtannotationfilepath);
	void save_txt(const string txtannotationfilepath);
	void drawannotation2Image(cv::Mat &src);
};
