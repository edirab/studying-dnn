### Custom YOLO v4 Object detection


- Основной репозиторий: https://github.com/AlexeyAB/darknet
- Для запуска необходимо только 3 файла: `*.weights`, `*.cfg` и текстовый файл с классами
- Описание параметров [cfg-файла по ссылке](https://github.com/AlexeyAB/darknet/wiki/CFG-Parameters-in-the-%5Bnet%5D-section)
- Запуск обучения `!./darknet detector train data/yolov4.data cfg/yolov4_custom_train.cfg yolov4.conv.137 -dont_show -map`

![](../imgs/Result.png)

![](../imgs/Dock.png)

![](../imgs/Chart_loss_dock.png)
### Изменения в оригинальном Colab

- В `yolov4_config.py`:
```
    classes=1
    max_batches=6000
    ...
    batch=8
    subdivisions=2
```

Соответственно, меняются файлы
```
	[INFO] Generating yolov4_custom_train.cfg successfully...
	[INFO] Generating yolov4_custom_test.cfg successfully...
```

- В `yolov4.data` указать `num_classes=1`, остальные пути как есть
- Создать свои файлы `test.txt` & `train.txt`
- Модифицирован `Makefile`: удалена строка, вызывающая ошибку компиляции

		ARCH= -gencode arch=compute_30,code=sm_30 \
		
   