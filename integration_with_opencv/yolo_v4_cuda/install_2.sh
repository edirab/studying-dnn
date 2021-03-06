echo "Compiling and linking project..."
g++ -Wall -o a.out\
	yolo_v4_cuda.cpp \
	-I/usr/local/include/opencv4 \
	-L/usr/local/lib/ \
	-pthread \
	-lopencv_bgsegm \
	-lopencv_calib3d \
	-lopencv_ccalib \
	-lopencv_core \
	-lopencv_cudaarithm \
	-lopencv_cudabgsegm \
	-lopencv_cudacodec \
	-lopencv_cudafeatures2d \
	-lopencv_cudafilters \
	-lopencv_cudaimgproc \
	-lopencv_cudalegacy \
	-lopencv_cudaobjdetect \
	-lopencv_cudaoptflow \
	-lopencv_cudastereo \
	-lopencv_cudawarping \
	-lopencv_cudev \
	-lopencv_dnn \
	-lopencv_dnn_objdetect \
	-lopencv_dnn_superres \
	-lopencv_dpm \
	-lopencv_features2d \
	-lopencv_flann \
	-lopencv_fuzzy \
	-lopencv_gapi \
	-lopencv_hfs \
	-lopencv_highgui \
	-lopencv_img_hash \
	-lopencv_imgcodecs \
	-lopencv_imgproc \
	-lopencv_intensity_transform \
	-lopencv_line_descriptor \
	-lopencv_mcc \
	-lopencv_ml \
	-lopencv_objdetect \
	-lopencv_optflow \
	-lopencv_saliency \
	-lopencv_shape \
	-lopencv_stitching \
	-lopencv_superres \
	-lopencv_surface_matching \
	-lopencv_text \
	-lopencv_video \
	-lopencv_videoio \
	-lopencv_xfeatures2d \
	-lopencv_ximgproc \
	-lopencv_xobjdetect