import cv2
import numpy as np

import time
# import threading from Thread
import threading

import sensor_data
import effects
from effects import No_filter as nof

done = 0
MAX_VALUE = 10000
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
    cap = cv2.VideoCapture(1)
    cap.set(3, 1920)
    cap.set(4, 1080)
    #cap.set(3, 1600)
    #cap.set(4, 900)
    #out = cv2.VideoWriter('vids/MulNoise.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (640,480))
    l_filter = [
                effects.Displacement_zoom_filter("../media/circle.png"),
                nof(),
                nof(),
                nof(),
                effects.Motion_blur_filter(),
                effects.Color_plane_filter(color=(0, 0, 0xFF)), #red
                effects.Color_plane_filter(color=(0, 0xFF, 0xFF)), #yellow
                effects.Color_plane_filter(), #pink / purple
                nof(),
                effects.Vertical_sin_effect(),
                effects.Mirror_effect(),
                nof(),
                effects.Displacement_zoom_filter("../media/square.png"),
                effects.Displacement_zoom_filter("../media/triangle.png"),
                nof(),
               ]
    r_filter = [
                effects.Circle_grad_filter(),
                nof(),
                nof(),
                effects.Border_filter(),
                nof(),
                effects.Duatone_filter(dua_layers=(0, 2), other_layer=(1, 0)), #red, blue
                effects.Duatone_filter(dua_layers=(1, 0), other_layer=(2, 0)), #yellow, blue
                effects.Duatone_filter(dua_layers=(1, 2), other_layer=(0, 1)), #purple, greenis
                effects.Pixelate_filter(),
                nof(),
                nof(),
                effects.RGB_shift_filter(),
                effects.Multiply_filter(),
                effects.Kaleidoscope_grad_filter(),
                effects.Negative_filter(),
               ]
    #r_filter = effects.Kaleidoscope_grad_filter()
    effect_number = 0
    osc_per_effect = 2
    osc_number = 0
    first_zero = True
    while (key_code != 27): #until esc key is pressed
        ret, frame = cap.read()
        print(frame.shape)
        val = sensor.values[3]
        intensity = abs(sensor.values[3])
        print(intensity)
        if (intensity > MAX_VALUE):
            intensity = MAX_VALUE
        intensity /= MAX_VALUE
        if (val > 0):
            r_filter[effect_number].set_intensity(intensity)
            r_filter[effect_number].apply_filter(frame)
            first_zero = True
        elif (val < 0):
            l_filter[effect_number].set_intensity(intensity)
            l_filter[effect_number].apply_filter(frame)
            first_zero = True
        else:
            if first_zero:
                osc_number = (osc_number + 1) % (osc_per_effect * 15)
                effect_number = osc_number // osc_per_effect
                r_filter[effect_number].reset()
                l_filter[effect_number].reset()
                first_zero = False
        cv2.imshow('pendulum', frame)
        #out.write(frame)
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
