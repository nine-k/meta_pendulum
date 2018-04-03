import serial
ser = serial.Serial('/dev/ttyACM0', 34800)

import cv2
SIZE = 10
import numpy as np
import time

key_code = 0
image = np.zeros([SIZE, SIZE, 3])
gyro_readings = [0] * 3
while (key_code != 27): #until esc key is pressed
    data = ser.read(20)
    data_start = 0
    for i in range(len(data)):
        if data[i] == data[i + 1] == data[i + 2] == 0:
            data_start = i + 3
            break
    for i in range(0, 3):
        gyro_readings[i] = int.from_bytes(data[data_start + 2 * i: data_start + 2 * (i + 1)],
                                       byteorder='big', signed = True)
    color_coeff = gyro_readings[0]
    print(color_coeff)
    print("test")
    image[:,:,0] = np.ones([SIZE, SIZE]) * (color_coeff > 0) * 64
    image[:,:,1] = np.ones([SIZE, SIZE]) * (color_coeff <= 0) * 64
    print("test1")
    cv2.imshow("test", image)
    key_code = cv2.waitKey(1)
