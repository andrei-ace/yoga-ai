# yoga-ai



#
./start_gpu.sh

cd /workspace/setup/vck5000/
source setup.sh

cd /workspace/models/AI-Model-Zoo
wget https://www.xilinx.com/bin/public/openDownload?filename=openpose_pruned_0_3-vck5000-DPUCVDX8H-r1.4.1.tar.gz -O openpose_pruned_0_3-vck5000-DPUCVDX8H-r1.4.1.tar.gz
sudo mkdir /usr/share/vitis_ai_library/models
tar -xzvf openpose_pruned_0_3-vck5000-DPUCVDX8H-r1.4.1.tar.gz
sudo cp openpose_pruned_0_3 /usr/share/vitis_ai_library/models -r

conda activate vitis-ai-tensorflow2
cd /workspace/yoga-ai/

sh build.sh


# Movie

ffmpeg -i file.webm -r 1/1 ./1/$filename%010d.bmp