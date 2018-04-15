import cv2
import numpy as np

import time
# import threading from Thread
import threading

import sensor_data
import effects

done = 0
sensor = sensor_data.Sensor()
cv2.setUseOptimized(True)

def serial_thread():
    global sensor
    global done
    while done == 0:
        sensor.parse_and_filter()

def video_thread():
    cv2.namedWindow("pendulum", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("pendulum", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    key_code = 0
    cap = cv2.VideoCapture(0)
    l_filter = effects.Pixelate_filter(20)
    r_filter = effects.Circle_grad_filter()
    while (key_code != 27): #until esc key is pressed
        ret, frame = cap.read()
        val = sensor.values[3]
        intensity = abs(sensor.values[3])
        if (intensity > 8000):
            intensity = 8000
        intensity /= 8000
        #intensity = np.arccos(intensity) / np.pi * 2
        print(intensity)
        if (val > 0):
            r_filter.angle_delta = intensity * 5 + 2
            r_filter.apply_filter(frame)
        elif (val < 0):
            l_filter.change_grain(int(intensity * 40))
            l_filter.apply_filter(frame)
        cv2.imshow('pendulum', frame)
        key_code = cv2.waitKey(1)
    cap.release()
    cv2.destroyAllWindows()
    global done
    done = 1

main_thread = threading.Thread(target=video_thread)#, args=("cv_thread"))
serial_thread = threading.Thread(target=serial_thread)#, args=("serial_thread"))
main_thread.start()
serial_thread.start()
main_thread.join()
print("done?")
