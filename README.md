# yoga-ai
[YouTube demo 3D pose article.xmodel](https://www.youtube.com/watch?v=z6NKuxKhq-E)

[YouTube demo 3D pose res.xmodel](https://www.youtube.com/watch?v=_IiREVrlGwo)

[YouTube demo 3D pose gan.xmodel](https://www.youtube.com/watch?v=5obPOOsSL-E)


## Clone this project inside the Vitis AI Library directory
```console
git clone git@github.com:andrei-ace/yoga-ai.git
```

## Build the docker image vitis-ai-gpu:1.4.1.978 
Follow the instructions from the Vitis AI project. VCK-5000-ES1 card is only supported by the 1.4.1 version.


## Add support for USB webcams and X11 
This is done by modifying the docker_run.sh
```console
docker_run_params=$(cat <<-END
    -v /dev/shm:/dev/shm \
    -v /opt/xilinx/dsa:/opt/xilinx/dsa \
    -v /opt/xilinx/overlaybins:/opt/xilinx/overlaybins \
    -v /etc/xbutler:/etc/xbutler \
    -e USER=$user -e UID=$uid -e GID=$gid \
    -e VERSION=$VERSION \
    -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v $HOME/.Xauthority:/tmp/.Xauthority \
    -v $DOCKER_RUN_DIR:/vitis_ai_home \
    -v $HERE:/workspace \
    -w /workspace \
    --device /dev/video0 \
    --rm \
    --network=host \
    --ipc=host \
    ${DETACHED} \
    ${RUN_MODE} \
    $IMAGE_NAME \
    $DEFAULT_COMMAND
END
)
```
## Start the Vitis AI docker container (GPU)
```console
./docker_run.sh xilinx/vitis-ai-gpu:1.4.1.978
```
## Run this inside docker to prepare the environment:
```console
cp /tmp/.Xauthority ~/
sudo chown vitis-ai-user:vitis-ai-group ~/.Xauthority
sudo usermod -a -G video $(whoami)
sudo su $(whoami)
cd /workspace/setup/vck5000/
source setup.sh
cd /workspace/models/AI-Model-Zoo
wget https://www.xilinx.com/bin/public/openDownload?filename=openpose_pruned_0_3-vck5000-DPUCVDX8H-r1.4.1.tar.gz -O openpose_pruned_0_3-vck5000-DPUCVDX8H-r1.4.1.tar.gz
sudo mkdir /usr/share/vitis_ai_library/models
tar -xzvf openpose_pruned_0_3-vck5000-DPUCVDX8H-r1.4.1.tar.gz
sudo cp openpose_pruned_0_3 /usr/share/vitis_ai_library/models -r
sudo usermod -a -G video vitis-ai-user
/usr/bin/pip3 install matplotlib
conda activate vitis-ai-tensorflow2
cd /workspace/yoga-ai/
sh build.sh
```
## Preparing the dataset
This project is using the annotation from [Human3.6M Dataset](https://drive.google.com/file/d/1ztokDig-Ayi8EYipGE1lchg5XlAoLmwY/view?usp=sharing)

The annotations.zip file has to be extracted inside the data/annotations/ directory

After extracting the zip content in data/annotations/ directory the data will be preprocessed and .tfrecords files will be created based on the ground truth from the Human3.6M Dataset.
```console
python prepare_data.py
```
Each 3D pose will be scaled (so that the distance from the head anchor to the hip anchor will be of 1 unit legth), centered to the hip anchor and randomly rotated. 
This script will create the following files:
```
ls -al ./data/Human36M_subject*
-rw-r--r-- 1 andrei andrei 12172996 mar 19 19:52 ./data/Human36M_subject11_joint_3d.tfrecords
-rw-r--r-- 1 andrei andrei 12977646 mar 19 19:47 ./data/Human36M_subject1_joint_3d.tfrecords
-rw-r--r-- 1 andrei andrei 20707511 mar 19 19:48 ./data/Human36M_subject5_joint_3d.tfrecords
-rw-r--r-- 1 andrei andrei 13055394 mar 19 19:49 ./data/Human36M_subject6_joint_3d.tfrecords
-rw-r--r-- 1 andrei andrei 21238789 mar 19 19:50 ./data/Human36M_subject7_joint_3d.tfrecords
-rw-r--r-- 1 andrei andrei 13517702 mar 19 19:51 ./data/Human36M_subject8_joint_3d.tfrecords
-rw-r--r-- 1 andrei andrei 16598153 mar 19 19:51 ./data/Human36M_subject9_joint_3d.tfrecords
```
## train
There are three DNN models:

1. A simple 4 layers fully connected DNN trainable with this command:
```console
python train-simple.py
```
2. The DNN described in the [A simple yet effective baseline for 3d human pose estimation](https://arxiv.org/pdf/1705.03098.pdf) article:
```console
python train-article.py
```
3. A DNN inspired by the previous model with fewer fully connected neurons but with more layers:
```console
python train-res.py
```
## quantize
TensorFlow2 and Vitis AI design flows are described in this [tuturial](https://github.com/Xilinx/Vitis-AI-Tutorials/tree/master/Design_Tutorials/08-tf2_flow)

Each model can be quantized using the following commands:
```console
python -u quantize.py --float_model model/simple/simple.h5 --quant_model model/simple/quant_simple.h5 --batchsize 64 --evaluate 2>&1 | tee quantize.log
```
```console
python -u quantize.py --float_model model/article/article.h5 --quant_model model/article/quant_article.h5 --batchsize 64 --evaluate 2>&1 | tee quantize.log
```
```console
python -u quantize.py --float_model model/residual/res.h5 --quant_model model/residual/quant_res.h5 --batchsize 64 --evaluate 2>&1 | tee quantize.log
```

## compile
The quantized model has to be compiled before using it
```console
vai_c_tensorflow2 --model model/simple/quant_simple.h5 --arch /opt/vitis_ai/compiler/arch/DPUCVDX8H/VCK5000/arch.json --output_dir model/simple --net_name simple
```
```console
vai_c_tensorflow2 --model model/article/quant_article.h5 --arch /opt/vitis_ai/compiler/arch/DPUCVDX8H/VCK5000/arch.json --output_dir model/article --net_name article
```
```console
vai_c_tensorflow2 --model model/residual/quant_res.h5 --arch /opt/vitis_ai/compiler/arch/DPUCVDX8H/VCK5000/arch.json --output_dir model/residual --net_name res
```

The model can be then viewed:
```console
xir png model/simple/simple.xmodel model/simple/simple.png
```
```console
xir png model/article/article.xmodel model/article/article.png
```
```console
xir png model/residual/res.xmodel model/residual/res.png
```

## run the 3D pose estimator on a picture:

## webcam demo 
This demo will showcase the 3D pose estimator on images captured in real time by a webcam
```console
sh build.sh
./build/yoga-ai-mt model/article/article.xmodel
```
# GAN

```console
mkdir -p ./data/video/frames
find ./data/video -maxdepth 1 -name '*.webm' -print0 | xargs -0 -i sh -c 'fullfile="{}"; filename=${fullfile##*/}; name=${filename%%.*}; ffmpeg -i "$fullfile" -r 10/1 ./data/video/frames/"$name"%010d.jpg'
```
```console
./build/yoga-ai-multiple ./data/video/frames > ./data/video/frames.json
```
```console
python prepare_data_gan.py ./data/video/

rm -rf logs/*
rm -rf ./model/gan/*
python train-gan.py

tensorboard --logdir logs/ --bind_all

python -u quantize.py --float_model model/gan/gan.h5 --quant_model model/gan/quant_gan.h5 --batchsize 64 --evaluate 2>&1 | tee quantize.log

vai_c_tensorflow2 --model model/gan/quant_gan.h5 --arch /opt/vitis_ai/compiler/arch/DPUCVDX8H/VCK5000/arch.json --output_dir model/gan --net_name gan
```
