import numpy as np
import cv2
from math import floor, ceil
import effects

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
    res_frame = frame
    if direction == 'r':
        res_frame = frame_mirror(res_frame)
    else:
        res_frame = frame_multiply(frame)[:,:640]
    res_frame = rgb_shift_frame(res_frame)
    return res_frame


out = cv2.VideoWriter('pixels.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (640,480))
cap = cv2.VideoCapture(0)
#frame_filter = effects.Ball_filter("../png/")
#frame_filter = effects.Motion_blur_filter(memory_window=20)
#frame_filter = effects.Displacement_mapping_filter("circle.png", (0,20))
#frame_filter = effects.Horizontal_distort_effect()
frame_filter = effects.Pixelate_grad_filter(1, 30)

while True:
    ret, frame = cap.read()
    frame = frame_filter.apply_filter(frame)
    cv2.imshow('frame', frame)
    out.write(frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
