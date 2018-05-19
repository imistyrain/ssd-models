#include "AnnotationFile.h"
#include "pugixml.hpp"
#include "fstream"
#include "mrutil.h"
#include "DataSetConfig.h"
using namespace std;
bool AnnotationFile::load_xml(const string annotationfilepath)
{
	pugi::xml_document doc;
	pugi::xml_parse_result result = doc.load_file(annotationfilepath.c_str());
	if (result.status != 0)
	{
		cout << result.description() << endl;
		return false;
	}
	auto annotationnode = doc.child("annotation");
	folder = annotationnode.child("folder").text().get();
	filename = annotationnode.child("filename").text().get();
	database = annotationnode.child("source").child("database").text().get();
	annotation = annotationnode.child("source").child("annotation").text().get();
	image = annotationnode.child("source").child("image").text().get();

	width = annotationnode.child("size").child("width").text().as_int();
	height = annotationnode.child("size").child("height").text().as_int();
	depth = annotationnode.child("size").child("depth").text().as_int();
	segmented = annotationnode.child("segmented").text().as_bool();
	auto objectsnode = annotationnode.select_nodes("object");
	for (auto it = objectsnode.begin(); it != objectsnode.end(); it++)
	{
		auto objectnode = (*it).node();
		Object object;
		object.name = objectnode.child("name").text().get();
		object.pose = objectnode.child("pose").text().get();
		auto bndboxnode = objectnode.child("bndbox");
		object.xmin = bndboxnode.child("xmin").text().as_int();
		object.ymin = bndboxnode.child("ymin").text().as_int();
		object.xmax = bndboxnode.child("xmax").text().as_int();
		object.ymax = bndboxnode.child("ymax").text().as_int();
		object.truncated = objectnode.child("truncated").text().as_bool();
		object.difficult = objectnode.child("difficult").text().as_bool();
		objects.push_back(object);
	}
	return true;
}
bool AnnotationFile::load_txt(const string annotationfilepath)
{
	ifstream fin(annotationfilepath);
	if (!fin)
		return false;
	int label;
	float xcenter, ycenter, wr, hr;
	objects.clear();
	while (!fin.eof())
	{
		fin >> label >> xcenter >> ycenter >> wr >> hr;
		Object object;
		object.name = int2string(label);
		object.xmin = (xcenter - 0.5*wr)*width;
		object.ymin = (ycenter - 0.5*hr)*height;
		object.xmax = (xcenter + 0.5*wr)*width;
		object.ymax = (ycenter + 0.5*hr)*height;
		objects.push_back(object);
	}
	return true;
}
void AnnotationFile::save_xml(const string xmlannotationfilepath)
{
	pugi::xml_document doc;
	pugi::xml_node rootnode=doc.append_child("annotation");
	rootnode.append_child("folder").text().set("3-1");
	rootnode.append_child("filename").text().set(filename.c_str());
	rootnode.append_child("path").text().set(filename.c_str());
	pugi::xml_node sourcenode = rootnode.append_child("source");
	sourcenode.append_child("database").text().set(database.c_str());
	sourcenode.append_child("annotation").text().set(annotation.c_str());
	pugi::xml_node sizenode = rootnode.append_child("size");
	sizenode.append_child("width").text().set(int2string(width).c_str());
	sizenode.append_child("height").text().set(int2string(height).c_str());
	sizenode.append_child("depth").text().set(int2string(depth).c_str());
	rootnode.append_child("segmented").text().set("0");
	for (auto object : objects)
	{
		pugi::xml_node objectnode = rootnode.append_child("object");
		objectnode.append_child("name").text().set(object.name.c_str());
		objectnode.append_child("pose").text().set(object.pose.c_str());
		objectnode.append_child("truncated").text().set(int(object.truncated));
		objectnode.append_child("difficult").text().set(int(object.difficult));
#if OUTPUT_OBJECT_POSSIBILITY
		objectnode.append_child("possibility").text().set(double2string(object.possibility).c_str());
#endif
		pugi::xml_node bndboxnode = objectnode.append_child("bndbox");
		bndboxnode.append_child("xmin").text().set(int2string(object.xmin).c_str());
		bndboxnode.append_child("ymin").text().set(int2string(object.ymin).c_str());
		bndboxnode.append_child("xmax").text().set(int2string(object.xmax).c_str());
		bndboxnode.append_child("ymax").text().set(int2string(object.ymax).c_str());
	}
	doc.save_file(xmlannotationfilepath.c_str());
}

map<string, int> AnnotationFile::labelmap = {};
void  AnnotationFile::set_labelmaps(const vector<string> classes)
{
	for (int i = 0; i < classes.size(); i++)
	{
		labelmap[classes[i]] = i;
	}
}

void AnnotationFile::save_txt(const string txtannotationfilepath)
{
	ofstream ftxt(txtannotationfilepath);
	for (auto object : objects)
	{
		ftxt << labelmap[object.name] << "\t" << (object.xmin + object.xmax)*0.5 / width << "\t"
			<< (object.ymin+object.ymax)*0.5 / height
			<< "\t" << (object.xmax-object.xmin)*1.0 / width << "\t" <<
			(object.ymax-object.ymin)*1.0 / height << endl;
	}
	ftxt.close();
}

void AnnotationFile::drawannotation2Image(cv::Mat &src)
{
	for (int i = 0; i < objects.size(); i++)
	{
		cv::Scalar color;
		if (objects[i].difficult)
			color = CV_RGB(255, 0, 0);
		else
			color = CV_RGB(0, 255, 0);
		rectangle(src, cv::Point(objects[i].xmin, objects[i].ymin), cv::Point(objects[i].xmax, objects[i].ymax), color);
		putText(src, objects[i].name, cv::Point(objects[i].xmin, objects[i].ymin), 3, 1, color);
	}
}
