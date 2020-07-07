# ncnn部署

## 转换模型到ncnn

修改[convert.sh](ncnn/convert.sh)中模型的路径
```
./convert.sh
```

## demo
首先需要编译[ncnn](https://github.com/Tencent/ncnn)
```
git clone https://github.com/Tencent/ncnn
cd ncnn
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=/usr/local ..
make -j4
make install
```
python端部署需要先编译和[pyncnn](https://github.com/caishanli/pyncnn)，方法和上述类似.s
```
cd ncnn
python3 demo.py
```
C++端
```
cd -
mkdir build
cd build
cmake ..
make -j4
./demo
```