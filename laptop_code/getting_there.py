import serial
ser = serial.Serial('/dev/ttyACM1', 38400, timeout=1.5)

import cv2
SIZE = 100
import numpy as np

import time
# import threading from Thread
import threading

gyro_readings = [0] * 3
done = 0

def frame_multiply(frame):
    small_frame = cv2.resize(frame, None, fx=1/6, fy=1/6, interpolation=cv2.INTER_AREA)
    res_frame = small_frame
    for i in range(5):
        res_frame = np.concatenate((res_frame, small_frame), axis=1)
    small_frame = res_frame
    for i in range(5):
        res_frame = np.concatenate((res_frame, small_frame), axis=0)
    return res_frame

def frame_mirror(frame):
    rows, cols, _ = frame.shape
    mid_frame = frame[:, cols // 4: 3 * cols // 4, :]
    mirrored_mid_frame = cv2.flip(mid_frame, -1)
    frame = np.concatenate((mid_frame, mirrored_mid_frame), axis=1)
    return frame

def rgb_shift_frame(frame):
    RSHIFT = 5
    GSHIFT = 5
    BSHIFT = 5
    frame[:,:,0] = np.pad(frame[:,:,0], ((0, 0), (RSHIFT, 0)), 'constant', constant_values=(0))[:,:-RSHIFT]
    frame[:,:,1] = np.pad(frame[:,:,1], ((GSHIFT, 0), (0, 0)), 'constant', constant_values=(0))[:-GSHIFT,:]
    frame[:,:,2] = np.pad(frame[:,:,2], ((0, BSHIFT), (0, 0)), 'constant', constant_values=(0))[BSHIFT:,:]
    return frame

def distort_frame(frame, direction):
    #frame = rgb_shift_frame(frame)
    if direction == 'r':
        res_frame = frame_mirror(frame)
    else:
        res_frame = frame_multiply(frame)
    res_frame = rgb_shift_frame(res_frame)
    return res_frame

def get_direction(vals):
    if (vals[0] > 0):
        return 'r'
    else:
        return 'l'

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
    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('frame', 1280, 960)
    key_code = 0
    cap = cv2.VideoCapture(0)
    while (key_code != 27): #until esc key is pressed
        ret, frame = cap.read()
        direction = get_direction(gyro_readings)
        frame = distort_frame(frame, direction)
        cv2.imshow('frame', frame)
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
