## Изучаем модуль dnn

- `yolo_v3` взято из [официального примера](https://docs.opencv.org/4.5.1/da/d9d/tutorial_dnn_yolo.html)
- `yolo_v4_cuda` - [отсюда](https://gist.github.com/YashasSamaga/e2b19a6807a13046e399f4bc3cca3a49)

NB: Экспериментальным путём установлено, что `cv::dnn::DNN_TARGET_CUDA` работает только в `Release`

### 1. `yolo_v4_cuda`

- Производительность **~ 17 FPS**, всё в одном потоке


### 2. `yolo_v3`

- Вручную захардкодили `preferrable target` && `preferrable backend`
- Производительность в среднем **27 FPS** засчёт захвата кадров в другом потоке
- Нельзя просто так взять и подставить `yolo4.weights` вместо 3-ей версии
- `scale` не меняем, всё перестаёт работать

- Запускаем

    `yolo_v3.exe --config=../../../yolov3.cfg --model=../../../yolov3.weights --classes=../../../object_detection_classes_yolov3.txt --width=416 --height=416 --scale=0.00392 --rgb`

    `yolo_v3.exe --config=../../../yolov3.cfg --model=../../../yolov3.weights --classes=../../../object_detection_classes_yolov3.txt --width=640 --height=480 --scale=0.00392 --rgb`	


