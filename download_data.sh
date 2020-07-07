#voc
download_voc(){
    root_dir = ${pwd}
    cd $root_dir}/data
    if [ ! -d voc ]; then
        mkdir voc
    fi
    cd voc
    if [ ! -d VOCdevkit ] ; then
        wget https://pjreddie.com/media/files/VOCtrainval_11-May-2012.tar
        wget https://pjreddie.com/media/files/VOCtrainval_06-Nov-2007.tar
        wget https://pjreddie.com/media/files/VOCtest_06-Nov-2007.tar
        tar xf VOCtrainval_11-May-2012.tar
        tar xf VOCtrainval_06-Nov-2007.tar
        tar xf VOCtest_06-Nov-2007.tar
    fi
    if [ ! -d lmdb ] ; then
        python ${root_dir}/python/create_lmdb.py
    fi
}
#face fddb
download_fddb(){
   #download from http://vis-www.cs.umass.edu/fddb/
    if [ ! -d lmdb] ; then
        python ${root_dir}/python/convert2lmdb.py fddb trainval
        python ${root_dir}/python/convert2lmdb.py fddb test
    fi
}
#hand
download_hand(){
    cd $root_dir}/data
    if [ ! -d Hand ] ; then
        mkdir Hand
    fi
    cd Hand
    if [ ! -d lmdb] ; then
        if [ ! -d training] ; then
            wget https://lmb.informatik.uni-freiburg.de/data/freihand/FreiHAND_pub_v2.zip
            unzip FreiHAND_pub_v2.zip
        fi
        python ${root_dir}/python/convert2lmdb.py Hand train
        python ${root_dir}/python/convert2lmdb.py Hand val
    fi
}
#Head
download_head(){
    cd $root_dir}/data
    if [ ! -d Head ] ; then
        mkdir Head
    fi
    cd Head
    #download yourself
    python ${root_dir}/python/convert2lmdb.py Head train
    python ${root_dir}/python/convert2lmdb.py Head val
}
#person 
download_widerperson(){
    cd $root_dir}/data
    if [ ! -d Person ] ; then
        mkdir Person
    fi
    cd Person
    #down file from from http://www.cbsr.ia.ac.cn/users/sfzhang/WiderPerson/
    python ${root_dir}/python/convert2lmdb.py Person trainval
    python ${root_dir}/python/convert2lmdb.py Person test
}
#insect
download_insect(){
    cd $root_dir}/data
    wget -c https://bj.bcebos.com/paddlex/datasets/insect_det.tar.gz
    tar zxvf insect_det.tar.gz
    mv insect_det insect
    python ${root_dir}/python/convert2lmdb.py insect train
    python ${root_dir}/python/convert2lmdb.py insect val
}

download_voc
download_fddb
#download_hand
#download_head
#download_widerperson