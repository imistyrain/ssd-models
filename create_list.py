import os,random,cv2

datasetId="fddb"
datasetyear=""
datasetname=datasetId+datasetyear
#rootdir="/home/data"
rootdir="D:/Face/Datasets/fddb"
sub_dir="ImageSets/Main"

def create_list():
    datasets=["trainval","test"]
    for dataset in datasets:
        dst_file=dataset+".txt"
        dataset_file=rootdir+"/"+sub_dir+"/"+dataset+".txt"
        with open(dataset_file) as fdataset:
            with open(dst_file,'w') as fdst:
                lines=fdataset.readlines()
                for line in lines:
                    filename=line[:-1]
                    imgfilepath=datasetname+"/images/"+filename+".jpg"
                    imglabelpath=datasetname+"/Annotations/"+filename+".xml"
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
                namesizetxt="test_name_size.txt"
                with open(namesizetxt,'w') as fsize:
                    lines=fdataset.readlines()
                    for line in lines:
                        imgfilepath=rootdir+"/images/"+line[:-1]+".jpg"
                        img=cv2.imread(imgfilepath)
                        line_size=line[:-1]+" "+str(img.shape[0])+" "+str(img.shape[1])+"\n"
                        fsize.write(line_size)

if __name__=="__main__":
    create_list()