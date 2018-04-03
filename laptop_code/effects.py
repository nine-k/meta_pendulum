import numpy as np
import cv2
import os
from datetime import datetime
from datetime import timedelta

class Ball_filter:
    def __init__(self, png_dir, fps=29,
                 dst_size = (640, 480)
                ):
        self.image_seq = []
        self.cur_frame = 0
        self.fps = fps
        self.frame_t_delta = 1 / self.fps * 10**6
        print(self.frame_t_delta)
        self.frame_change_time = datetime.now()
        directory = os.fsencode(png_dir)
        for frame in os.listdir(directory):
            frame_name = os.fsdecode(frame)
            if frame_name.endswith(".png"):
                tmp = cv2.imread(png_dir + frame_name)
                tmp = cv2.resize(tmp, dst_size, interpolation=cv2.INTER_AREA)
                self.image_seq.append(tmp)
        self.frame_total = len(self.image_seq)


    def apply_filter(self, frame, position=(100, 100)):
        cur_image = self.image_seq[self.cur_frame]
        overlay_mask = cv2.cvtColor(cur_image, cv2.COLOR_BGR2GRAY)
        res, overlay_mask = cv2.threshold(overlay_mask, 10, 1, cv2.THRESH_BINARY_INV)

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
