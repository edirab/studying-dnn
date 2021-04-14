echo "Compiling and linking project..."
g++ -Wall -o a.out\
	yolo_v4_cuda.cpp \
	-I/usr/local/include/opencv4 \
	-L/usr/local/lib/ \
	-pthread \
	-lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_imgcodecs \
	-lopencv_videoio -lopencv_dnn -lopencv_dnn_objdetect \
	-lopencv_cudaarithm -opnecv_cudacodec -lopencv_cudaimgproc -lopencv_cudalegacy \
	-lopencv_cudaobjdetect -lopencv_cudawarping -lopencv_cudev