import os
import ncnn
import cv2

this_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),"../")
class Detector:
    def __init__(self, target_size=160, num_threads=1, use_gpu=False):
        self.target_size = target_size
        self.num_threads = num_threads
        self.use_gpu = use_gpu

        self.mean_vals = [127.5, 127.5, 127.5]
        self.norm_vals = [0.007843, 0.007843, 0.007843]

        self.net = ncnn.Net()
        self.net.opt.use_vulkan_compute = self.use_gpu

        self.net.load_param(this_dir+"ncnn/mask.param")
        self.net.load_model(this_dir+"ncnn/mask.bin")

        self.class_names = ["face","mask"]
            
    def __del__(self):
        self.net = None

    def __call__(self, img):
        mat_in = ncnn.Mat.from_pixels_resize(img, ncnn.Mat.PixelType.PIXEL_BGR, img.shape[1], img.shape[0], self.target_size, self.target_size)
        mat_in.substract_mean_normalize(self.mean_vals, self.norm_vals)

        ex = self.net.create_extractor()
        ex.set_num_threads(self.num_threads)

        ex.input("data", mat_in)

        mat_out = ncnn.Mat()
        ex.extract("detection_out", mat_out)

        objects = []

        #printf("%d %d %d\n", mat_out.w, mat_out.h, mat_out.c)
        for i in range(mat_out.h):
            values = mat_out.row(i)
            objects.append(values)
        return objects

    def drawDetection(self, img, detections):
        for i in range(len(detections)):
            det = detections[i]
            label = self.class_names[int(det[0])-1]+"_"+str(det[1])
            x = int(det[2]*img.shape[1])
            y = int(det[3]*img.shape[0])
            x2 = int(det[4]*img.shape[1])
            y2 = int(det[5]*img.shape[0])
            cv2.rectangle(img,(x,y),(x2,y2),(255,0,0))
            cv2.putText(img,label,(x,y),1,1,(0,0,255))
        return img

def test_dir(detector, dir=this_dir+"/images"):
    files = os.listdir(dir)
    for file in files:
        print(file)
        imgpath = dir+"/"+file
        img = cv2.imread(imgpath)
        if img is None:
            continue
        detections = detector(img)
        img = detector.drawDetection(img, detections)
        #cv2.imshow("img",img)
        #cv2.waitKey()
        cv2.imwrite(this_dir+"/output/"+file, img)

def main():
    detector = Detector()
    test_dir(detector)

if __name__=="__main__":
    main()