#coding=utf-8
import os
import random
import cv2
import subprocess
import shutil
import platform
#生成训练所需的列表文件和lmdb文件
#注意按序号修改数据集名称、年份以及数据集的位置和caffe安装位置以及标签文件
datasetname="voc"
modeldir="models"
if platform.system()=="Windows":
    caffe_root="D:/CNN/ssd"#1
    data_dir="E:/Face/Datasets/"+datasetname+"/"#2
else:
    caffe_root=os.path.expanduser("~")+"/Detection/ssd"
data_dir="data/"+datasetname+"/"
label_map_file=datasetname+"/labelmap.prototxt"#3
dbdir=datasetname+"/lmdb"

def create_list():
    datasets=["trainval","test"]
    for dataset in datasets:
        dst_file=modeldir+"/"+dataset+".txt"
        print("creating "+dst_file)
        dataset_file=data_dir+"/ImageSets/Main"+"/"+dataset+".txt"
        with open(dataset_file) as fdataset:
            with open(dst_file,'w') as fdst:
                lines=fdataset.readlines()
                for line in lines:
                    filename=line.split()[0]
                    imgfilepath="images/"+filename+".jpg"
                    imglabelpath="Annotations/"+filename+".xml"
                    img_line=imgfilepath+" "+imglabelpath+"\n"
                    fdst.write(img_line)
        if dataset=="trainval":
            lines=[]
            with open(dst_file) as fdst:
                lines=fdst.readlines()
                random.shuffle(lines)
            with open(dst_file,'w') as fdst:
                for line in lines:
                    fdst.write(line)
        if dataset=="test":
            with open(dataset_file) as fdataset:
                namesizetxt=modeldir+"/"+"test_name_size.txt"
                with open(namesizetxt,'w') as fsize:
                    lines=fdataset.readlines()
                    for line in lines:
                        imgfilepath=data_dir+"/images/"+line.split()[0]+".jpg"
                        #print(imgfilepath)
                        img=cv2.imread(imgfilepath)
                        line_size=line.split()[0]+" "+str(img.shape[0])+" "+str(img.shape[1])+"\n"
                        fsize.write(line_size)

def create_data():
    sets=["trainval","test"]
    for set in sets:
        list_file=modeldir+"/"+set+".txt"
        db=dbdir+"/"+set+"_lmdb"
        cmd = "{}/build/tools/convert_annoset" \
            " --anno_type=detection" \
            " --label_map_file={}" \
            " --encode_type=jpg --encoded"\
            " {} {} {}" \
            .format(caffe_root,label_map_file,data_dir,list_file,db)
        print(cmd)
        process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
        process.communicate()[0]

if __name__=="__main__":
    if os.path.exists(dbdir):
        subdirs=os.listdir(dbdir)
        for sub in subdirs:
            subdir=dbdir+"/"+sub
            files=os.listdir(subdir)
            for file in files:
                filepath=subdir+"/"+file
                os.remove(filepath)
            os.rmdir(subdir)
    else:
        os.makedirs(dbdir)
    if not os.path.exists(modeldir):
        os.makedirs(modeldir)
    create_list()
    create_data()