### Новый эксперимент по измерению производительности и точности нейросети

- Зафиксируем гиперпараметры
```
	constexpr float CONFIDENCE_THRESHOLD = 0.8;
	constexpr float NMS_THRESHOLD = 0.05;
```
- Порядок следования классов во всех экспериментах: dock, w.c., b.c.

- Видео с одиночным белым маркером `white_circle.mp4`:


```
	Total frames: 900
			Time difference = 44144449[╡s]
			Time difference = 44144[ms]
			Time difference = 44[s]
	objs:    0   1   2   3   4   5   6   7   8   9  10  11  12  13  14
	dock:  397 485  18   0   0   0   0   0   0   0   0   0   0   0   0
	w.c.:   14 883   3   0   0   0   0   0   0   0   0   0   0   0   0
	b.c.:  900   0   0   0   0   0   0   0   0   0   0   0   0   0   0
```


- Видео с одиночным чёрным маркером `black_circle.mp4`:

```
	Total frames: 932
			Time difference = 45638048[╡s]
			Time difference = 45638[ms]
			Time difference = 45[s]
	objs:    0   1   2   3   4   5   6   7   8   9  10  11  12  13  14
	dock:  450 480   2   0   0   0   0   0   0   0   0   0   0   0   0
	w.c.:  932   0   0   0   0   0   0   0   0   0   0   0   0   0   0
	b.c.:   10 920   2   0   0   0   0   0   0   0   0   0   0   0   0
```

