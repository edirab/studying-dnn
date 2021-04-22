Измеряем точность и производительность нейронки

Video 1 (with) 915 кадров всего

Frame #0
	   dock:
			1 (751, 253, 451, 280)
	   w.c.:
			1 (865, 500, 58, 64)
			4 (1045, 496, 51, 65)
			6 (1401, 313, 188, 99)
	   b.c.:
			4 (1033, 320, 62, 49)
			1 (861, 330, 54, 50)
			

### Для видео с объектом:

Total frames: 915
        Found pairs of black circles: 709 0,77
        Found pairs of white circles: 797 0,87
		
dock: 152 760 3
white 122 709 84
black 107 797 11

N1 = 915
P = 915
N = 0

TP_1 = 760
TP_2 = 709
TP_3 = 797

Т.к. на каждом кадре есть объект
TN_1 = 0
TN_2 = 0
TN_3 = 0

FN_1 = 152 + 3
FN_2 = 122 + 84
FN_3 = 107 + 11

FP_1,2,3 = 0

### Для видео без объекта:

Total frames: 904
        Found pairs of white circles: 0 0
        Found pairs of black circles: 11 0

dock: 842 62 0
white 904 0 0
black 893 11 0

N2 = 904

P = 0
N = 904

TP_1, 2, 3 = 0

FN_1 = 62
FN_2 = 
FN_3 = 
