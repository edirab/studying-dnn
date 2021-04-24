### Новый эксперимент

```
	constexpr float CONFIDENCE_THRESHOLD = 0.8;
	constexpr float NMS_THRESHOLD = 0.05;
```

- Видео с одиночным белым маркером `white_circle.mp4`:
- Порядок следования классов: dock, w.c., b.c.

```
	Total frames: 900
			Found pairs of white circles: 3 0
			Found pairs of black circles: 0 0
	397 485 18
  **14 883 3**
	900 0 0
```

Кадров где 
- 0 объектов: 14
- 1 объект: 883
- 2 объекта и более: 3


- Видео с одиночным чёрным маркером:

```
	Total frames: 932
			Found pairs of white circles: 0 0
			Found pairs of black circles: 2 0
	450 480 2
	932 0 0
  **10 920 2**
```

