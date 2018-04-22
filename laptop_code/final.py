#! /usr/bin/python3
import cv2
import numpy as np

import time
# import threading from Thread
import threading

import sensor_data
import effects
from effects import No_filter as nof

from datetime import datetime
from datetime import timedelta

done = 0
MAX_VALUE = 10000
sensor = sensor_data.Sensor(offsets=(0,0,0,0,0,8700))
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
    cap.set(3, 1920)
    cap.set(4, 1080)
    #cap.set(3, 1600)
    #cap.set(4, 900)
    #out = cv2.VideoWriter('vids/MulNoise.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (640,480))
    l_filter = [
                effects.Displacement_zoom_filter("../media/triangle.png"),
                effects.Color_plane_filter(color=(0, 0, 0xFF)), #red
                effects.Color_plane_filter(color=(0, 0xFF, 0xFF)), #yellow
                effects.Color_plane_filter(), #pink / purple
                effects.Displacement_zoom_filter("../media/square.png"),
                effects.Vertical_sin_effect(),
                effects.Mirror_effect(),
                effects.Displacement_zoom_filter("../media/circle.png"),
                effects.Displacement_zoom_filter("../media/square_pattern.png"), #square pattern
                effects.Motion_blur_filter(),
                effects.Border_filter(),
                #effects.Displacement_zoom_filter("../media/triangle_pattern.png"), #triag
               ]
    r_filter = [
                effects.Rotate_grad_filter("../media/triangle.png", angle_delta=120, time_delta=10**5),
                effects.Pixelate_grad_filter(),
                effects.Horizontal_distort_effect(),
                effects.Duatone_filter(dua_layers=(0, 2), other_layer=(1, 0), threshold=100), #red, blue
                effects.Rotate_grad_filter("../media/square.png", angle_delta=90, time_delta=10**5),
                effects.RGB_shift_filter(),
                effects.Png_overlay_filter("../videoplayback3/"),
                effects.Rotate_grad_filter("../media/circle.png"),
                effects.Multiply_filter(),
                #effects.Kaleidoscope8_filter(),
                effects.Kaleidoscope_filter(HOR=True, VERT=True),#4
                effects.Kaleidoscope_filter(HOR=False, VERT=True)#2
               ]
    idle_filter = effects.Png_overlay_filter("../idle_png/", fps=1)
    #r_filter = effects.Kaleidoscope_grad_filter()
    effect_number = 0
    osc_per_effect = 2
    osc_number = 0
    first_zero = True
    ZERO_THRESH = 400
    COUNTER_THRESH = 200
    dir_left = True
    idle_time_start = datetime.now()
    while (key_code != 27): #until esc key is pressed
        #ret, frame = cap.read()
        frame = cap.read()[1][180:,160:160+1600,:]
        frame[:,:,:] = cv2.flip(frame, 1) #VER FLIP
        if frame.shape != (1920, 1080, 3):
            frame = cv2.resize(frame, effects.DEFAULT_SIZE, interpolation=cv2.INTER_AREA)
        val = sensor.values[5]
        #print(val)
        intensity = 1
        if (val > ZERO_THRESH):
            r_filter[effect_number].set_intensity(intensity)
            r_filter[effect_number].apply_filter(frame)
            dir_left = True
            first_zero = True
        elif (val < -ZERO_THRESH):
            if dir_left:
                #effect_number = np.random.randint(0, len(l_filter))
                effect_number = (effect_number + 1) % len(l_filter)
                r_filter[effect_number].reset()
                l_filter[effect_number].reset()
                dir_left = False
                first_zero = False
            l_filter[effect_number].set_intensity(intensity)
            l_filter[effect_number].apply_filter(frame)
            dir_left = False
            first_zero = True
        else:
            if dir_left:
                #effect_number = np.random.randint(0, len(l_filter))
                effect_number = (effect_number + 1) % len(l_filter)
                r_filter[effect_number].reset()
                l_filter[effect_number].reset()
                dir_left = False
            if first_zero:
                idle_time_start = datetime.now()
                first_zero = False
            else:
                time_diff = datetime.now() - idle_time_start
                if time_diff.seconds > 5:
                    idle_filter.apply_filter(frame)
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
