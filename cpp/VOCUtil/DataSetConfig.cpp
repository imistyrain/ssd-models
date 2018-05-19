#include "DataSetConfig.h"
#include "pugixml.hpp"
#include "iostream"
#include "fstream"
#include "mrdir.h"
#include "random"
#include <chrono>
#include "AnnotationFile.h"
int DatasetConfig::load_file(const string configpath)
{
	pugi::xml_document doc;
	doc.load_file(configpath.c_str());
	pugi::xml_node rootnode = doc.child("dataset");
	datasetname = rootnode.child("name").text().get();
	year = rootnode.child("year").text().get();
	imagedir = rootnode.child("imagedir").text().get();
	annotationdir = rootnode.child("annotationdir").text().get();
	labelsdir = rootnode.child("labelsdir").text().get();

	currentlabelingclass = rootnode.child("currentlabelingclass").text().get();
	lastlabeledindex = rootnode.child("lastlabeledindex").text().as_int();
	bsavexml = rootnode.child("bsavexml").text().as_bool();
	bsavetxt = rootnode.child("bsavetxt").text().as_bool();
	pugi::xml_node classesnode = rootnode.child("classes");
	classes.clear();

	for (auto it = classesnode.first_child(); it; it = it.next_sibling())
	{
		classes.push_back(it.text().get());
	}
	AnnotationFile::set_labelmaps(classes);
	return 0;
}
void DatasetConfig::save_file(const string configpath)
{
	pugi::xml_document doc;
	pugi::xml_node rootnode = doc.append_child("dataset");
	rootnode.append_child("name").text().set(datasetname.c_str());
	rootnode.append_child("year").text().set(year.c_str());	
	rootnode.append_child("imagedir").text().set(imagedir.c_str());
	rootnode.append_child("annotationdir").text().set(annotationdir.c_str());
	rootnode.append_child("labelsdir").text().set(labelsdir.c_str());
	rootnode.append_child("currentlabelingclass").text().set(currentlabelingclass.c_str());
	rootnode.append_child("lastlabeledindex").text().set(lastlabeledindex);
	rootnode.append_child("bsavexml").text().set(bsavexml);
	rootnode.append_child("bsavetxt").text().set(bsavetxt);
	pugi::xml_node classesnode = rootnode.append_child("classes");
	for (int i = 0; i < classes.size(); i++)
	{
		classesnode.append_child("class").text().set(classes[i].c_str());
	}
	doc.save_file(configpath.c_str());
}

void DatasetConfig::init(const string dir)
{
	rootdir = dir.substr(0,dir.rfind("/"));
	datasetdir = dir;
	std::string configpath = dir + "/" + configfile;
	load(configpath);
}
void DatasetConfig::load(const string configpath)
{
	if (EXISTS(configpath.c_str()))
		load_file(configpath);
}

void DatasetConfig::initvoc()
{
	datasetname = "VOC";
	year = "0712";
	lastlabeledindex = 0;
	currentlabelingclass = "car";
	annotationdir = "Annotations";
	imagedir = "JPEGImages";
	labelsdir = "labels";
	string classnames[] = { "aeroplane", "bicycle", "bird",
		"boat", "bottle", "bus", "car", "cat", "chair", "cow",
		"diningtable", "dog", "horse", "motorbike", "person",
		"pottedplant", "sheep", "sofa", "train", "tvmonitor" };
	bsavexml = true;
	bsavetxt = true;
	classes.clear();
	for (int i = 0; i < 20; i++)
	{
		classes.push_back(classnames[i]);
	}
}

void DatasetConfig::initWithNames(const std::vector<std::string>&objnames)
{
	lastlabeledindex = 0;
	currentlabelingclass = objnames[lastlabeledindex];
	annotationdir = "Annotations";
	imagedir = "images";
	labelsdir = "labels";
	bsavexml = true;
	bsavetxt = true;
	classes.clear();
	for (int i = 0; i < objnames.size(); i++)
	{
		classes.push_back(objnames[i]);
	}
}

int DatasetConfig::generatetrainvaltxt(const string datasetprefix, const float trainratio, const float valratio, const float testratio)
{
	std::vector<std::vector<std::string>>filebylabels;
	for (int i = 0; i < AnnotationFile::labelmap.size(); i++)
	{
		std::vector<std::string>files1label;
		filebylabels.push_back(files1label);
	}
	std::string imgdir = datasetdir + "/" + imagedir;
	auto files = getAllFilesinDir(imgdir);
	for (int i = 0; i < files.size(); i++)
	{
		AnnotationFile af;
		std::string annopath = datasetdir + "/" + annotationdir + "/" + files[i];
		annopath = annopath.substr(0, annopath.length() - 3) + "xml";
		if (af.load_xml(annopath))
		{
			for (int j = 0; j < af.objects.size(); j++)
			{
				int label = AnnotationFile::labelmap[af.objects[j].name];
				filebylabels[label].push_back(files[i]);
			}
		}
	}
	if (bsavetxt)
	{
		ofstream ftrainval(datasetdir + "/trainval.txt");
		ofstream ftest(datasetdir + "/test.txt");
		for (int i = 0; i < filebylabels.size(); i++)
		{
			auto file1label = filebylabels[i];
			unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
			shuffle(file1label.begin(), file1label.end(), std::default_random_engine(seed));
			for (int j = 0; j < file1label.size(); j++)
			{
				string filepath = datasetprefix + datasetname + "/images/" + file1label[j];
				if (j < (trainratio + valratio)*file1label.size())
				{
					ftrainval << filepath << endl;
				}
				else if (j < (trainratio + valratio + testratio)*file1label.size())
				{
					ftest << filepath << endl;
				}
			}
		}
		ftrainval.close();
		ftest.close();
	}
	if (bsavexml)
	{
		std::string imagesetsdir = datasetdir + "/" + "ImageSets";
		if (!EXISTS(imagesetsdir.c_str()))
		{
			MKDIR(imagesetsdir.c_str());
		}
		std::string maindir = imagesetsdir + "/" + "Main";
		if (!EXISTS(maindir.c_str()))
		{
			MKDIR(maindir.c_str());
		}
		ofstream ftrain(maindir + "/train.txt");
		ofstream fval(maindir + "/val.txt");
		ofstream ftrainval(maindir + "/trainval.txt");
		ofstream ftest(maindir + "/test.txt");
		for (int i = 0; i < filebylabels.size(); i++)
		{
			auto file1label = filebylabels[i];
			unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
			shuffle(file1label.begin(), file1label.end(), std::default_random_engine(seed));
			for (int j = 0; j < file1label.size(); j++)
			{
				string filepath = file1label[j];
				filepath = filepath.substr(0, filepath.length() - 4);
				if (j < trainratio*file1label.size())
				{
					ftrain << filepath << endl;
					ftrainval << filepath << endl;
				}
				else if (j < (trainratio + valratio)*file1label.size())
				{
					fval << filepath << endl;
					ftrainval << filepath << endl;
				}
				else if (j < (trainratio + valratio + testratio)*file1label.size())
				{
					ftest << filepath << endl;
				}
			}
		}
		ftrain.close();
		fval.close();
		ftrainval.close();
		ftest.close();
	}
	return 0;
}