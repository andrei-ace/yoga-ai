result=0 && pkg-config --list-all | grep opencv4 && result=1
if [ $result -eq 1 ]; then
	OPENCV_FLAGS=$(pkg-config --cflags --libs-only-L opencv4)
else
	OPENCV_FLAGS=$(pkg-config --cflags --libs-only-L opencv)
fi

CXX=${CXX:-g++}
for file in $(ls src/*.cpp); do
	filename=${file%.*}
	filename=${filename#src\/}
	
	$CXX -std=c++17 -O2 -I$PWD/src/common -I. $PWD/src/common/common.cpp $PWD/src/common/pose.cpp \
	-o build/${filename} ${file} \
	-I/usr/include/python3.6m -lpython3.6m \
	-lvitis_ai_library-openpose \
	-lvitis_ai_library-dpu_task \
	-lvitis_ai_library-xnnpp \
	-lvitis_ai_library-model_config \
	-lvitis_ai_library-math \
	-lvart-util \
	-lxir \
	-pthread \
	-ljson-c \
	-lglog \
	-lvart-runner \
	${OPENCV_FLAGS} \
	-lopencv_core \
	-lopencv_videoio \
	-lopencv_imgproc \
	-lopencv_imgcodecs \
	-lopencv_calib3d \
	-lopencv_objdetect \
	-lopencv_features2d \
	-lopencv_highgui
done