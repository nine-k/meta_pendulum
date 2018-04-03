import serial
ser = serial.Serial('/dev/ttyACM0', 38400, timeout=1.5)

import cv2
SIZE = 100
import numpy as np

import time
# import threading from Thread
import threading 

gyro_readings = [0] * 3
done = 0

def serial_thread():
    global gyro_readings
    global done
    while done == 0:
        zero_count = 0
        while zero_count < 3 and done == 0:
            cur_byte = ser.read(1)[0]
            if cur_byte == 0:
                zero_count += 1
            else:
                zero_count = 0
        for i in range(0, 3):
            ser.read(1) #separating byte 0xFF
            data = ser.read(2)
            gyro_readings[i] = int.from_bytes(data, byteorder='big', signed = True)
        # time.sleep(0.02)
        print(gyro_readings)

def video_thread():
    key_code = 0
    image = np.zeros([SIZE, SIZE, 3])
    global done
    while (key_code != 27): #until esc key is pressed
        color_coeff = gyro_readings[0]
        # print(color_coeff)
        image[:,:,0] = np.ones([SIZE, SIZE]) * (color_coeff > 0) * 64
        image[:,:,1] = np.ones([SIZE, SIZE]) * (color_coeff <= 0) * 64
        cv2.imshow("test", image)
        key_code = cv2.waitKey(1)
    done = 1

main_thread = threading.Thread(target=video_thread)#, args=("cv_thread"))
serial_thread = threading.Thread(target=serial_thread)#, args=("serial_thread"))
main_thread.start()
serial_thread.start()
main_thread.join()
print("done?")
