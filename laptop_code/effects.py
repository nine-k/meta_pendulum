import numpy as np
import cv2
import os
from datetime import datetime
from datetime import timedelta
from collections import deque

class Ball_filter:
    def __init__(self, png_dir, fps=29,
                 dst_size = (640, 480)
                ):
        self.image_seq = []
        self.image_overlay = []
        self.cur_frame = 0
        self.fps = fps
        self.frame_t_delta = 1 / self.fps * 10**6
        print(self.frame_t_delta)
        self.frame_change_time = datetime.now()
        directory = os.fsencode(png_dir)
        file_names = []
        for frame in os.listdir(directory):
            frame_name = os.fsdecode(frame)
            if frame_name.endswith(".png"):
                file_names.append(frame_name)
        for frame_name in sorted(file_names):
            tmp = cv2.imread(png_dir + frame_name)
            tmp = cv2.resize(tmp, dst_size, interpolation=cv2.INTER_AREA)
            overlay_mask = cv2.cvtColor(tmp, cv2.COLOR_BGR2GRAY)
            _, overlay_mask = cv2.threshold(overlay_mask, 10, 1, cv2.THRESH_BINARY_INV)
            self.image_seq.append(tmp)
            self.image_overlay.append(overlay_mask)
        self.frame_total = len(self.image_seq)
        print(self.frame_total)



    def apply_filter(self, frame, position=(100, 100)):
        cur_image = self.image_seq[self.cur_frame]
        overlay_mask = self.image_overlay[self.cur_frame]

        h, w = overlay_mask.shape
        overlay_mask = np.repeat(overlay_mask, 3).reshape((h, w, 3))

        frame[:h, :w,] *= overlay_mask
        frame += cur_image
        time_dif = (datetime.now() - self.frame_change_time).microseconds
        #print(time_dif)
        if (time_dif >= self.frame_t_delta):
            self.cur_frame = (self.cur_frame + 1) % self.frame_total
            if (self.cur_frame == 0):
                print("done")
            self.frame_change_time = datetime.now()
        return frame

class Motion_blur_filter:
    def __init__(self, memory_window=10):
        self.memory = list()
        self.memory_window = memory_window

    def apply_filter(self, frame):
        self.memory.append(frame)
        if len(self.memory) > self.memory_window:
            self.memory.pop(0)
        new_frame = np.zeros(frame.shape, dtype="uint8")#np.sum(self.memory, axis=2) / len(self.memory)
        for fr in self.memory:
            print(len(self.memory))
            new_frame += fr * 3 // len(self.memory) // 4
        new_frame += frame // 4
        print(frame.shape, new_frame.shape)
        return new_frame
