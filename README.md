# yoga-ai

#
```console
./start_gpu.sh
```
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
# 1. tfrecord
```console
python prepare_data.py
```
# 2. train
```console
python train-simple.py
```
# 3. quantize
```console
python -u quantize.py --float_model model/simple/simple.h5 --quant_model model/simple/quant_simple.h5 --batchsize 64 --evaluate 2>&1 | tee quantize.log
```
```console
python -u quantize.py --float_model model/article/article.h5 --quant_model model/article/quant_article.h5 --batchsize 64 --evaluate 2>&1 | tee quantize.log
```
```console
python -u quantize.py --float_model model/residual/res.h5 --quant_model model/residual/quant_res.h5 --batchsize 64 --evaluate 2>&1 | tee quantize.log
```

# 4. compile
```console
vai_c_tensorflow2 --model model/simple/quant_simple.h5 --arch /opt/vitis_ai/compiler/arch/DPUCVDX8H/VCK5000/arch.json --output_dir model/simple --net_name simple
```
```console
vai_c_tensorflow2 --model model/article/quant_article.h5 --arch /opt/vitis_ai/compiler/arch/DPUCVDX8H/VCK5000/arch.json --output_dir model/article --net_name article
```
```console
vai_c_tensorflow2 --model model/residual/quant_res.h5 --arch /opt/vitis_ai/compiler/arch/DPUCVDX8H/VCK5000/arch.json --output_dir model/residual --net_name res
```

# 5
```console
xir png model/simple/simple.xmodel model/simple/simple.png
```
```console
xir png model/article/article.xmodel model/article/article.png
```
```console
xir png model/residual/res.xmodel model/residual/res.png
```

# 6 
```console
./build/yoga-ai-mt model/article/article.xmodel
```
# Movie

```console
mkdir -p ./data/video/mov1
ffmpeg -i ./data/video/mov1.webm -vf scale=640:-1 -r 1/1 ./data/video/mov1/$filename%010d.jpg
```
```console
./build/yoga-ai-multiple ./data/video/mov1 > ./data/video/mov1.json
```
```console
python prepare_data_gan.py ./data/video/

python train-gan.py

python -u quantize.py --float_model model/gan/gan.h5 --quant_model model/gan/quant_gan.h5 --batchsize 64 --evaluate 2>&1 | tee quantize.log

vai_c_tensorflow2 --model model/gan/quant_gan.h5 --arch /opt/vitis_ai/compiler/arch/DPUCVDX8H/VCK5000/arch.json --output_dir model/gan --net_name gan
```