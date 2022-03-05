# yoga-ai



#
./start_gpu.sh

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
conda activate vitis-ai-tensorflow2
cd /workspace/yoga-ai/
sh build.sh

# 1. tfrecord
# 2. train
# 3. quantize
python -u quantize.py --float_model model/simple/simple.h5 --quant_model model/simple/quant_simple.h5 --batchsize 64 --evaluate 2>&1 | tee quantize.log

python -u quantize.py --float_model model/residual/res.h5 --quant_model model/residual/quant_res.h5 --batchsize 64 --evaluate 2>&1 | tee quantize.log

# 4. compile
vai_c_tensorflow2 --model model/simple/quant_simple.h5 --arch /opt/vitis_ai/compiler/arch/DPUCVDX8H/VCK5000/arch.json --output_dir model/simple --net_name simple
vai_c_tensorflow2 --model model/article/quant_article.h5 --arch /opt/vitis_ai/compiler/arch/DPUCVDX8H/VCK5000/arch.json --output_dir model/article --net_name article
vai_c_tensorflow2 --model model/residual/quant_res.h5 --arch /opt/vitis_ai/compiler/arch/DPUCVDX8H/VCK5000/arch.json --output_dir model/residual --net_name res


# 5
xir png model/simple/simple.xmodel model/simple/simple.png
xir png model/article/article.xmodel model/article/article.png
xir png model/residual/res.xmodel model/residual/res.png

# Movie

ffmpeg -i mov1.webm -vf scale=640:-1 -r 1/1 ./mov1/$filename%010d.jpg
./run_open_pose_multiple ./data/video/mov1 > ./data/annotations/mov1.json
python prepare_data_gan.py