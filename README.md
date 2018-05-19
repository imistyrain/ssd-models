# ssd-face

## fddb库上使用ssd训练的人脸检测器

如果你正在使用opencv3.3及以上的版本，而且想直接看结果而不是训练的话，可以直接跳过第一节的caffe-ssd配置

![](https://i.imgur.com/j9AMJnb.png)

### 1. 下载并编译[ssd](https://github.com/weiliu89/caffe)

```
git clone https://github.com/weiliu89/caffe
cd caffe
git checkout ssd
cp Makefile.config.example Makefile.config
make -j8
make py
```

为方便起见，后文将下载的caffe所在文件夹记为$SSD_ROOT

### 2. 下载本项目

切换到$SSD_ROOT/data/目录下

```Shell
cd data
git clone https://github.com/imistyrain/ssd-face
```

### 3. 运行demo

下载预训练好的[模型](http://pan.baidu.com/s/1c1A7ESC)（120000 iters, 约90M, 密码：4iyb），将其置于cpp文件夹下，确保结构如图下图所示
![](https://i.imgur.com/UH7wTPh.png)

#### 3.1 python版本

```python
python demo.py
```

#### 3.2 Windows下命令行版本

```
RunFaceDetect.bat
```

#### 3.3 C++版本

如果你正在使用opencv3.3及以上版本，其自带了caffe支持,如果是3.2及以下版本，则需要外加opencv_extra重新编译opencv,(注意勾选WITH_DNN)

双击打开ssd-face.sln将SSDFace设为启动项，编译完成后运行即可

###### Note:其中opencv跨平台自动化配置可参见[MRHead](https://github.com/imistyrain/MRHead)

### 4. 训练自己的数据

#### 4.1 准备训练数据，将数据转换为VOC格式

如果想直接训练fddb的话，可以直接下载已经转换好的[fddb库](http://pan.baidu.com/s/1pK8jglP)(百度网盘， 密码：g33x，约102M)，并将其置于/home/data/Face2017下，这个步骤过程可以参见[将fddb标注转换为VOC格式标注](http://blog.csdn.net/minstyrain/article/details/77938596)

当然，你也可以换成自己的数据，推荐一个好用的标注工具:[MRLabeler](https://github.com/imistyrain/MRLabeler)

#### 4.2 生成训练所需格式数据

```
python create_all.py
```
其中create_list.py把训练图片路径及其标注按行写入到trainval.txt中，把测试图片路径及其标注按行写入到test.txt中，把测试图片路径及其大小（高度、宽度）写入到test_name_size.txt中

```
Note:由于fddb中含有多级目录,为了兼容SSD及YOLO的训练结构要求,此脚本将路径中的"/"转换为了"_"
```

create_data.sh用于生成训练所需的lmdb文件,由于要支持多标签的输入，因此其内部使用了slice data layer，避免使用hdf5生成文件过大的问题

#### 4.3 启动训练

```
python train.py
```

## 参考

* [用SSD训练自己的数据集(VOC2007格式)](http://blog.csdn.net/zhy8623080/article/details/73188594)

* [将fddb标注转换为VOC格式标注](http://blog.csdn.net/minstyrain/article/details/77938596)

* [yolo-face](https://github.com/imistyrain/yolo-face)