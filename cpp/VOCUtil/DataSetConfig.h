#pragma once
#include "string"
#include "vector"
using namespace std;
class DatasetConfig
{
public:
	DatasetConfig()
	{
	}
	void init(const string dir = "./");
	void load(const string configpath);
	void initvoc();
	void initWithNames(const std::vector<std::string>&objnames);
	int load_file(const string configpath);
	void save_file(const string configpath);
	int generatetrainvaltxt(const string datasetprefix="/home/yanhe/data/", const float trainratio = 0.7, const float valratio = 0.2, const float testratio = 0.1);
	string rootdir="./";
	string datasetdir;
	string configfile = "mrconfig.xml";
	string datasetname="MRDatasets";
	string year="2017";
	string imagedir = "images";
	string annotationdir="Annotations";
	string labelsdir = "labels";	
	int lastlabeledindex=0;
	string currentlabelingclass;
	vector<string>classes;
	bool bsavexml=true;
	bool bsavetxt=true;
private:
	string intermediatedir = "intermediate";
};
