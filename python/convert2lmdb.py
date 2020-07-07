# convert all other datasets to lmdb
import os
import cv2
import shutil
import random
import numpy as np
import caffe
import lmdb
import json
import argparse
from caffe.proto import caffe_pb2
import xml.etree.ElementTree as ET
from tqdm import tqdm

CLASSES={
    "Face":["face"],
    "fddb":["face"],
    'wider':['face'],
    "Mask": ['face','face_mask'],
    "Head":["head"],
    "Person":["pedestrians"],
    "Hand":["hand"],
    "Car":["car"],
    "tower":["tower"],
    "insect":["leconte","boerner","armandi","linnaeus","coleoptera","acuminatus"],
    "voc":  ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse','motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train','tvmonitor']
}

class Registry(object):
    def __init__(self):
        self._module_dict = dict()
    def _register_module(self, name, func):
        module_name = name
        if module_name in self._module_dict:
            raise KeyError('{} is already registered'.format(module_name))
        self._module_dict[module_name] = func
    def register_module(self, name, cls):
        self._register_module(name, cls)
        return cls
    def get(self, key):
        return self._module_dict.get(key, None)

#conver anno to datum
# img, [h,w,c] uint8 image using cv2.imread
# bboxes, N*5 labels, each is [xmin,ymin,xmax,ymax,label], with label index from 0
def anno2datum(img,bboxes):
    if len(bboxes) == 0:
        return
    annotated_datum = caffe_pb2.AnnotatedDatum()
    annotated_datum.type = annotated_datum.BBOX
    datum = annotated_datum.datum 
    datum.channels = img.shape[2]
    datum.width = img.shape[1]
    datum.height = img.shape[0]
    datum.encoded = True
    datum.label = -1
    datum.data = cv2.imencode('.jpg',img)[1].tobytes()
    groups = annotated_datum.annotation_group
    for bbox in bboxes:
        found_group = False
        instance_id = 0
        label = int(bbox[4]) + 1 # background is 0
        for group in groups:
            if group.group_label == label:
                if len(group.annotation) == 0:
                    instance_id = 0
                else:
                    instance_id = len(group.annotation)
                found_group= True
                annotation = group.annotation.add()
                break
        if not found_group:
            group = groups.add()
            instance_id = 0
            group.group_label = label
            annotation = group.annotation.add()
        annotation.instance_id = instance_id
        annotation.bbox.xmin = bbox[0] * 1.0 /img.shape[1]
        annotation.bbox.ymin = bbox[1] * 1.0 /img.shape[0]
        annotation.bbox.xmax = bbox[2] * 1.0 /img.shape[1]
        annotation.bbox.ymax = bbox[3] * 1.0 /img.shape[0]
    return annotated_datum

#for voc xml annotation
# data/voc
#   --images
#   --Annotations
#   --train.txt
#   --val.txt
# each line in *.txt only contain filename like 000001.jpg
def xml2lmdb(args):
    data_dir="data/"+args.dataset
    lmdb_root = data_dir+"/lmdb"
    if not os.path.exists(lmdb_root):
        os.makedirs(lmdb_root)
    lmdb_dir = lmdb_root+"/"+args.split+"_lmdb"
    if os.path.exists(lmdb_dir):
        shutil.rmtree(lmdb_dir)
    db = lmdb.open(lmdb_dir,map_size=1e10)
    with db.begin(write=True) as txn:
        listfile_path = data_dir+"/"+args.split+".txt"
        if not os.path.exists(listfile_path):
            listfile_path = data_dir+"/ImageSets/Main"+"/"+args.split+".txt"
        with open(listfile_path) as f:
            cat2label = {cat: i for i, cat in enumerate(CLASSES[args.dataset])}
            lines=f.readlines()
            for line in tqdm(lines):
                filename=line.split()[0]
                if filename.endswith("jpg"):
                    filepath=data_dir+"/images/"+filename
                    xml_path=data_dir+"/Annotations/"+filename[:-4]+".xml"
                else:
                    filepath=data_dir+"/images/"+filename+".jpg"
                    xml_path=data_dir+"/Annotations/"+filename+".xml"
                img = cv2.imread(filepath)
                if img is None:
                    print(filepath+" cannot read")
                    continue
                if not os.path.exists(xml_path):
                    print(xml_path+" has no annotation")
                    continue
                tree = ET.parse(xml_path)
                root = tree.getroot()
                bboxes = []
                for obj in root.findall('object'):
                    name = obj.find('name').text
                    if name not in CLASSES[args.dataset]:
                        print(filepath+" has no expect label "+name)
                        continue
                    label = cat2label[name]
                    difficult = int(obj.find('difficult').text)
                    if difficult:
                        continue
                    bbox = obj.find('bndbox')
                    x = float(bbox.find('xmin').text)
                    y = float(bbox.find('ymin').text)
                    x2 = float(bbox.find('xmax').text)
                    y2 = float(bbox.find('ymax').text)
                    bbox = [x,y,x2,y2,label]
                    bboxes.append(bbox)
                if len(bboxes) == 0:
                    continue
                annotated_datum = anno2datum(img, bboxes)
                txn.put(filename,annotated_datum.SerializeToString())
#for data from paddledetection
# data/insect
#   --JPEGImages
#   --Annotations
#   --train_list.txt
#   --val_list.txt
#   --test_list.txt
# each line in *.txt contain filepath like JPEGImages/0001.jpg Annotations/0001.xml
def paddle2lmdb(args):
    data_dir="data/"+args.dataset
    lmdb_root = data_dir+"/lmdb"
    if not os.path.exists(lmdb_root):
        os.makedirs(lmdb_root)
    lmdb_dir = lmdb_root+"/"+args.split+"_lmdb"
    if os.path.exists(lmdb_dir):
        shutil.rmtree(lmdb_dir)
    db = lmdb.open(lmdb_dir,map_size=1e10)
    with db.begin(write=True) as txn:
        labelpath = data_dir+"/"+args.split+"_list.txt"
        with open(labelpath) as f:
            cat2label = {cat: i for i, cat in enumerate(CLASSES[args.dataset])}
            lines = f.readlines()
            for line in tqdm(lines):
                line = line.strip()
                filename = line.split(" ")[0]
                imgpath = data_dir+"/"+filename
                xml_path = data_dir+"/"+line.split(" ")[1]
                img = cv2.imread(imgpath)
                if img is None:
                    print(imgpath+" cannot read")
                    continue
                if not os.path.exists(xml_path):
                    print(xml_path+" has no annotation")
                    continue
                tree = ET.parse(xml_path)
                root = tree.getroot()
                bboxes = []
                for obj in root.findall('object'):
                    name = obj.find('name').text
                    if name not in CLASSES[args.dataset]:
                        print(imgpath+" has no expect label "+name)
                        continue
                    label = cat2label[name]
                    difficult = int(obj.find('difficult').text)
                    if difficult:
                        continue
                    bbox = obj.find('bndbox')
                    x = float(bbox.find('xmin').text)
                    y = float(bbox.find('ymin').text)
                    x2 = float(bbox.find('xmax').text)
                    y2 = float(bbox.find('ymax').text)
                    bbox = [x,y,x2,y2,label]
                    bboxes.append(bbox)
                if len(bboxes) == 0:
                    continue
                annotated_datum = anno2datum(img, bboxes)
                txn.put(filename,annotated_datum.SerializeToString())

#txt annotation
# data/***
#   --images
#   --train.txt
#   --val.txt
# each line: imgpath,xmin,ymin,xmax,ymax, label
# label index from 1
def txt2lmdb(args):
    data_dir="data/"+args.dataset
    lmdb_root = data_dir+"/lmdb"
    if not os.path.exists(lmdb_root):
        os.makedirs(lmdb_root)
    lmdb_dir = lmdb_root+"/"+args.split+"_lmdb"
    if os.path.exists(lmdb_dir):
        shutil.rmtree(lmdb_dir)
    db = lmdb.open(lmdb_dir,map_size=1e10)
    with db.begin(write=True) as txn:
        annopath = data_dir+"/"+args.split+".txt"
        with open(annopath) as f:
            lines = f.readlines()
            for line in tqdm(lines):
                items = line.strip().split(" ")
                filename = items[0]
                imgpath = data_dir+"/images/"+filename
                img = cv2.imread(imgpath)
                if img is None:
                    print("cannot read "+imgpath)
                    continue
                bboxes = []
                labels = items[1].split(",")
                for i in range(len(labels)/5):
                    xmin = float(labels[i*5])
                    ymin = float(labels[i*5+1])
                    xmax = float(labels[i*5+2])
                    ymax = float(labels[i*5+3])
                    label = int(labels[i*5+4])
                    bboxes.append([xmin,ymin,xmax,ymax,label-1])
                if len(bboxes) == 0:
                    continue
                annotated_datum = anno2datum(img, bboxes)
                txn.put(filename,annotated_datum.SerializeToString())

#for brainwash head dataset
# data/Head
#   --brainwash_10_27_2014_images
#   --brainwash_11_13_2014_images
#   --brainwash_11_24_2014_images
#   --brainwash_train.idl
#   --brainwash_val.idl
#   --brainwash_test.idl
def idl2lmdb(args):
    data_dir="data/"+args.dataset
    lmdb_root = data_dir+"/lmdb"
    if not os.path.exists(lmdb_root):
        os.makedirs(lmdb_root)
    lmdb_dir = lmdb_root+"/"+args.split+"_lmdb"
    if os.path.exists(lmdb_dir):
        shutil.rmtree(lmdb_dir)   
    db = lmdb.open(lmdb_dir,map_size=1e10)
    with db.begin(write=True) as txn:
        anno_file=data_dir+"/brainwash_"+args.split+".idl"
        with open(anno_file) as f:
            lines = f.readlines()
            for line in tqdm(lines):
                items = line[:-1].split(":")
                if len(items)==2:
                    imgpath = items[0][1:-1]
                    imgname = imgpath.split("/")[-1]
                    img = cv2.imread(data_dir+"/"+imgpath)
                    if img is None:
                        print(imgpath+" cannot read")
                        continue
                    items = items[1][1:-1].replace(",","")
                    items = items.replace("(","")
                    items = items.replace(")","")
                    items = items.split(" ")
                    items = [int(float(b)) for b in items]
                    bboxes = []
                    for i in range(int(len(items)/4)):
                        x = items[4*i]
                        y = items[4*i+1]
                        x2 = items[4*i+2]
                        y2 = items[4*i+3]
                        bboxes.append([x,y,x2,y2,0])
                    if len(bboxes) == 0:
                        continue
                    annotated_datum = anno2datum(img, bboxes)
                    txn.put(imgname,annotated_datum.SerializeToString())

#for coco with json annotation dataset
def convertjson(txn,json_path):
    with open(json_path) as f:
        samples = json.load(f)
        images_dir = os.path.dirname(json_path)+"/../images"
        for sample in tqdm(samples):
            imagename = sample["file_name"]
            imgpath = images_dir+"/"+imagename
            img = cv2.imread(imgpath)
            h,w,_ = img.shape
            if img is None:
                print(imagename + "Not found")
                continue
            bboxes = []
            objs = sample["object"]
            for obj in objs:
                bbox = obj['bbox']
                bbox[0] = bbox[0]
                bbox[1] = bbox[1]
                bbox[2] = bbox[0]+bbox[2]
                bbox[3] = bbox[1]+bbox[3]
                bbox.append(0)
                bboxes.append(bbox)
            if len(bboxes) == 0:
                continue
            annotated_datum = anno2datum(img, bboxes)
            txn.put(imagename.encode(),annotated_datum.SerializeToString())
def coco2lmdb(args):
    data_dir="data/"+args.dataset
    lmdb_root = data_dir+"/lmdb"
    lmdb_dir = lmdb_root+"/"+args.split+"_lmdb"
    if os.path.exists(lmdb_dir):
        shutil.rmtree(lmdb_dir)
    if not os.path.exists(lmdb_root):
        os.makedirs(lmdb_root)
    db = lmdb.open(lmdb_dir,map_size=1e10)
    with db.begin(write=True) as txn:
        json_path =  "data/"+args.dataset+"/annotations/instances_"+args.split+".json"
        convertjson(txn,json_path)

def freihand2lmdb(args):
    data_dir="data/"+args.dataset
    lmdb_root = data_dir+"/lmdb"
    lmdb_dir = lmdb_root+"/"+args.split+"_lmdb"
    if os.path.exists(lmdb_dir):
        shutil.rmtree(lmdb_dir)
    if not os.path.exists(lmdb_root):
        os.makedirs(lmdb_root)
    db = lmdb.open(lmdb_dir,map_size=1e10)
    with db.begin(write=True) as txn:
        anno_file=data_dir+"/annotations/freihand_"+args.split+".json"
        with open(anno_file) as f:
            data = json.load(f)
            for anno in tqdm(data['annotations']):
                filename = "{:08d}".format(anno['image_id'])+".jpg"
                img = cv2.imread(data_dir+"/training/rgb/"+filename)
                if img is None:
                    print(filename+"not found")
                    continue
                bboxes = []
                bbox = anno['bbox']
                bbox[2] += bbox[0]
                bbox[3] += bbox[1]
                bbox.append(0)
                bboxes.append(bbox)
                annotated_datum = anno2datum(img, bboxes)
                txn.put(filename,annotated_datum.SerializeToString())
#for bdd100k
# data/Car
#   --images
#   --labels
#   --train.txt
#   --val.txt
# each line contain imgpath like 100k/train/61c0de9c-996cae66.jpg
def bdd2lmdb(args):
    data_dir="data/"+args.dataset
    lmdb_root = data_dir+"/lmdb"
    lmdb_dir = lmdb_root+"/"+args.split+"_lmdb"
    if os.path.exists(lmdb_dir):
        shutil.rmtree(lmdb_dir)
    if not os.path.exists(lmdb_root):
        os.makedirs(lmdb_root)
    db = lmdb.open(lmdb_dir,map_size=1e10)
    with db.begin(write=True) as txn:
        valfile = data_dir+"/"+args.split+".txt"
        with open(valfile) as f:
            lines = f.readlines()
            for line in tqdm(lines):
                imgpath = data_dir+"/images/"+line.strip()
                img = cv2.imread(imgpath)
                annopath = data_dir+"/labels/"+line[:-4]+"json"
                with open(annopath) as fanno:
                    data = json.load(fanno)
                    objs = data['frames'][0]['objects']
                    bboxes = []
                    for obj in objs:
                        label = obj['category']
                        if 'box2d' in obj and label == "car":     
                            bbox = obj['box2d']
                            x1 = float(bbox['x1'])
                            y1 = float(bbox['y1'])
                            x2 = float(bbox['x2'])
                            y2 = float(bbox['y2'])
                            bboxes.append([x1,y1,x2,y2,0]) 
                    if len(bboxes) == 0:
                        print(imgpath + " has no valid size ")
                        continue
                    annotated_datum = anno2datum(img, bboxes)
                    txn.put(imgpath,annotated_datum.SerializeToString()) 
#for widerface
# data/Face
#   --WIDER_train
#   --WIDER_val
#   --wider_face_split
#       --wider_face_train_bbx_gt.txt
#       --wider_face_val_bbx_gt.txt
def wider2lmdb(args, min_size = 30):
    import sys
    data_dir="data/"+args.dataset
    lmdb_root = data_dir+"/lmdb"
    lmdb_dir = lmdb_root+"/"+args.split+"_lmdb"
    if os.path.exists(lmdb_dir):
        shutil.rmtree(lmdb_dir)
    if not os.path.exists(lmdb_root):
        os.makedirs(lmdb_root)
    db = lmdb.open(lmdb_dir, map_size=1e10)
    with db.begin(write=True) as txn:
        annopath = data_dir+"/wider_face_split/wider_face_"+args.split+"_bbx_gt.txt"
        imgdir = data_dir+"/WIDER_"+args.split+"/images"
        with open(annopath) as f:
            while(True):
                imgpath = f.readline()[:-1]
                sys.stdout.write("\r"+imgpath)
                if imgpath == "":
                    break
                img = cv2.imread(imgdir+"/"+imgpath)
                numbbox=int(f.readline())
                bboxes = []
                for _ in range(numbbox):
                    line = f.readline()
                    line = line.split()
                    line = [int(l) for l in line]
                    size = max(line[2],line[3])
                    bbox = line[:4]
                    bbox[2] += bbox[0]
                    bbox[3] += bbox[1]
                    bbox.append(0)
                    if size <= min_size:
                        continue
                    bboxes.append(bbox)
                if len(bboxes) == 0:
                    print(imgpath + " has no valid size ")
                    continue
                annotated_datum = anno2datum(img, bboxes)
                txn.put(imgpath.encode(),annotated_datum.SerializeToString())

def mask2lmdb(args, min_size = 20):
    data_dir="data/"+args.dataset
    lmdb_root = data_dir+"/lmdb"
    lmdb_dir = lmdb_root+"/"+args.split+"_lmdb"
    if os.path.exists(lmdb_dir):
        shutil.rmtree(lmdb_dir)
    if not os.path.exists(lmdb_root):
        os.makedirs(lmdb_root)
    db = lmdb.open(lmdb_dir,map_size=1e10)
    with db.begin(write=True) as txn:
        dir = data_dir+'/'+args.split
        files = os.listdir(dir)
        files = [ f for f in files if f.endswith('.xml')]
        if args.split.find("train"):
            files = random.shuffle(files)
        cat2label = {cat: i for i, cat in enumerate(CLASSES[args.dataset])}
        for file in tqdm(files):
            xml_path = dir+'/'+file
            tree = ET.parse(xml_path)
            root = tree.getroot()
            filename = file.replace('xml','jpg')
            imgpath = dir+'/'+filename
            img = cv2.imread(imgpath)
            if img is None:
                print("cannot read "+imgpath)
                continue
            bboxes = []
            for obj in root.findall('object'):
                name = obj.find('name').text
                if name not in CLASSES[args.dataset]:
                    print(imgpath+" has no expect label "+name)
                    continue
                label = cat2label[name]
                bbox = obj.find('bndbox')
                x = float(bbox.find('xmin').text)
                y = float(bbox.find('ymin').text)
                x2 = float(bbox.find('xmax').text)
                y2 = float(bbox.find('ymax').text)
                bbox = [x,y,x2,y2,label]
                bboxes.append(bbox)
            if len(bboxes) == 0:
                continue
            annotated_datum = anno2datum(img, bboxes)
            txn.put(filename.encode(),annotated_datum.SerializeToString())

def lmdb2image(args, show=False, gen_anchors=True,normalized=True):
    data_dir ="data/"+args.dataset
    lmdb_dir = data_dir+"/lmdb/"+args.split+"_lmdb"
    if not os.path.exists(lmdb_dir):
        print(lmdb_dir+" not exists")
        return
    db = lmdb.open(lmdb_dir)
    txn = db.begin()
    cursor = txn.cursor()
    annotated_datum = caffe_pb2.AnnotatedDatum()
    if gen_anchors:
        data = []
    labels = CLASSES[args.dataset]
    statics = len(labels)*[0]
    index = 0
    num_images = txn.stat()['entries']
    pbar = tqdm(range(num_images))
    for key, value in cursor:
        pbar.set_description("{}/{}".format(index,num_images))
        pbar.update(1)
        index += 1
        annotated_datum.ParseFromString(value)
        groups = annotated_datum.annotation_group
        #print(len(groups))
        if show or not normalized:
            datum = annotated_datum.datum
            img = np.fromstring(datum.data,dtype=np.uint8)
            img = cv2.imdecode(img,-1)       
            height, width, _ = img.shape
        for group in groups:
            for annotation in group.annotation:
                bbox = annotation.bbox
                if bbox.xmax-bbox.xmin<=0 or bbox.ymax-bbox.ymin<=0:
                        continue
                labelindex = group.group_label-1
                label = labels[labelindex]+"_"+str(annotation.instance_id)
                statics[labelindex] += 1
                if show or not normalized:
                    x1 = int(bbox.xmin*width)
                    y1 = int(bbox.ymin*height)
                    x2 = int(bbox.xmax*width)
                    y2 = int(bbox.ymax*height)
                    cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,0))
                    cv2.putText(img,label,(x1,y1),3,1,(0,0,255))
                if gen_anchors:
                    if normalized:
                        data.append([bbox.xmax-bbox.xmin,bbox.ymax-bbox.ymin])
                    else:
                        data.append([x2-x1,y2-y1])
        if args.savegt:
            filename=key.decode().replace("/","_")
            cv2.imwrite("output/gt/"+filename,img)
        if show:
            cv2.putText(img,key.decode(),(0,20),3,1,(0,0,255))
            cv2.imshow("img",img)
            cv2.waitKey()
    total = 0
    for i,st in enumerate(statics):
        total += st
        print(labels[i]+": "+str(st))
    print("-------Total: "+str(total))
    if gen_anchors:
        from get_anchors import get_anchors
        get_anchors(data)

funcs = Registry()
funcs.register_module("voc",xml2lmdb)
funcs.register_module("fddb",xml2lmdb)
funcs.register_module("wider",wider2lmdb)
funcs.register_module("Face",xml2lmdb)
funcs.register_module("Mask",mask2lmdb)
funcs.register_module("Person",xml2lmdb)
funcs.register_module("Head",idl2lmdb)
funcs.register_module("Hand",freihand2lmdb)
funcs.register_module("Car",bdd2lmdb)
funcs.register_module("tower",txt2lmdb)
funcs.register_module("insect",paddle2lmdb)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default="Mask")
    parser.add_argument('--split', default="val")
    parser.add_argument('--savegt', default=False)
    return parser.parse_args()

if __name__=="__main__":
    args = get_args()
    func = funcs.get(args.dataset)
    func(args)
    lmdb2image(args)