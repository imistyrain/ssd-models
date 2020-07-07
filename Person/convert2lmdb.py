import os
import cv2
import shutil
import numpy as np
import caffe
import lmdb
from caffe.proto import caffe_pb2
import xml.etree.ElementTree as ET
from tqdm import tqdm

DATASET="Person"
CLASSES=["background","pedestrians"]

data_dir="data/"+DATASET+"/"

def voc2lmdb(split="train"):
    lmdb_dir = DATASET+"/lmdb/"+split+"_lmdb"
    if os.path.exists(lmdb_dir):
        shutil.rmtree(lmdb_dir)
    db = lmdb.open(lmdb_dir,map_size=1e10)
    with db.begin(write=True) as txn:
        dataset_file=data_dir+split+".txt"
        with open(dataset_file) as f:
            lines = f.readlines()
            cat2label = {cat: i for i, cat in enumerate(CLASSES)}
            for line in tqdm(lines):
                filename=line.strip()
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
                annotated_datum = caffe_pb2.AnnotatedDatum()
                annotated_datum.type = annotated_datum.BBOX
                datum = annotated_datum.datum 
                datum.channels = img.shape[2]
                datum.width = img.shape[0]
                datum.height = img.shape[1]
                datum.encoded = True
                datum.label = -1
                datum.data = cv2.imencode('.jpg',img)[1].tobytes()
                groups = annotated_datum.annotation_group
                instance_id = 0
                for obj in root.findall('object'):
                    name = obj.find('name').text
                    if name not in CLASSES:
                        #print(filepath+" has no expect label "+name)
                        continue
                    label = cat2label[name]
                    bbox = obj.find('bndbox')
                    group = groups.add()
                    group.group_label=label
                    annotation = group.annotation.add()
                    annotation.instance_id = instance_id
                    annotation.bbox.xmin = float(bbox.find('xmin').text)/img.shape[0]
                    annotation.bbox.ymin = float(bbox.find('ymin').text)/img.shape[1]
                    annotation.bbox.xmax = float(bbox.find('xmax').text)/img.shape[0]
                    annotation.bbox.ymax = float(bbox.find('ymax').text)/img.shape[1]
                    instance_id += 1
                txn.put(filename,annotated_datum.SerializeToString())

def lmdb2image(split="train",save=False):
    lmdb_dir = DATASET+"/lmdb/"+split+"_lmdb"
    db = lmdb.open(lmdb_dir)
    txn = db.begin()
    cursor = txn.cursor()
    annotated_datum = caffe_pb2.AnnotatedDatum()
    for key, value in cursor:
        annotated_datum.ParseFromString(value)
        datum = annotated_datum.datum
        groups = annotated_datum.annotation_group
        img = np.fromstring(datum.data,dtype=np.uint8)
        img = cv2.imdecode(img,-1)       
        cv2.putText(img,key,(0,20),3,1,(0,0,255))
        for group in groups:
            for annotation in group.annotation:       
                x1 = int(annotation.bbox.xmin*datum.width)
                y1 = int(annotation.bbox.ymin*datum.height)
                x2 = int(annotation.bbox.xmax*datum.width)
                y2 = int(annotation.bbox.ymax*datum.height)
                label = CLASSES[group.group_label]+"_"+str(annotation.instance_id)
                cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,0))
                cv2.putText(img,label,(x1,y1),3,1,(0,0,255))
        if save:
            filename=key.replace("/","_")
            cv2.imwrite("output/gt/"+filename,img)
        cv2.imshow("img",img)
        cv2.waitKey()

def main():
    voc2lmdb()
    voc2lmdb("val")
    lmdb2image()

if __name__=="__main__":
    main()