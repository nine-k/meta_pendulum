import numpy as np
import cv2
import os
from datetime import datetime
from datetime import timedelta
from collections import deque

DEFAULT_SIZE = (640, 480)

class Png_overlay_filter:
    def __init__(self, png_dir, fps=29,
                 dst_size = DEFAULT_SIZE
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
            tmp = cv2.resize(tmp, frame_size, interpolation=cv2.INTER_AREA)
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
    def __init__(self, memory_window=10, alpha=1):
        self.memory = deque()
        self.memory_window = memory_window
        self.memory_frame = None
        self.alpha = alpha

    def apply_filter(self, frame):
        if self.memory_frame is None:
            self.memory_frame = np.zeros(frame.shape, dtype="double")
        self.memory.append(frame)
        self.memory_frame += (frame / self.memory_window)
        if len(self.memory) > self.memory_window:
            print(len(self.memory))
            self.memory_frame -= (self.memory[0] / self.memory_window)
            self.memory.popleft()
        return (self.memory_frame * self.alpha + frame * (1 - self.alpha)).astype("uint8")

class Displacement_mapping_filter:
    def __init__(self, template_name, offset, frame_size=DEFAULT_SIZE):
        tmp = cv2.imread(template_name)
        tmp = cv2.resize(tmp, frame_size, interpolation=cv2.INTER_AREA)

        overlay_mask = cv2.cvtColor(tmp, cv2.COLOR_BGR2GRAY)

        _, self.template_mask = cv2.threshold(overlay_mask, 10, 1, cv2.THRESH_BINARY_INV)
        self.template_mask = np.repeat(self.template_mask, 3).reshape((frame_size[1], frame_size[0], 3))

        _, self.template_mask_inv = cv2.threshold(overlay_mask, 10, 1, cv2.THRESH_BINARY)
        self.template_mask_inv = np.roll(self.template_mask_inv, offset, (0, 1))
        self.template_mask_inv = np.repeat(self.template_mask_inv, 3).reshape((frame_size[1], frame_size[0], 3))

        self.offset = offset

    def apply_filter(self, frame):
        template_frame = frame * self.template_mask
        template_frame = np.roll(template_frame, self.offset, (0, 1))
        frame *= self.template_mask_inv
        return template_frame + frame

class Horizontal_distort_effect:
    def __init__(self, lo=0, hi=10, frame_size=DEFAULT_SIZE):
        self.offsets = np.random.randint(lo, hi, frame_size[1], "uint8")
        self.lo = lo
        self.hi = hi
        self.frame_size = frame_size

    def apply_filter(self, frame):
        self.offsets = np.random.randint(self.lo, self.hi, self.frame_size[1], "uint8")
        for row in range(len(self.offsets)):
            frame[row, :, :] = np.roll(frame[row, :, :], self.offsets[row], 0)
        return frame

class Wind_distort_effect:
    def __init__(self, lo=-10, hi=20, frame_size=DEFAULT_SIZE):
        self.offsets = np.random.randint(lo, hi, frame_size[1], "uint8")

    def apply_filter(self, frame):
        for row in range(len(self.offsets)):
            frame[row, :, :] = np.roll(frame[row, :, :], self.offsets[row], 0)
        return frame

class Pixelate_filter:
    def __init__(self, pixel_size, fr_size = DEFAULT_SIZE):
        self.pixel_size = pixel_size
        self.fr_size = fr_size

    def change_grain(sz):
        self.pixel_size = sz

    def apply_filter(self, frame):
        frame = cv2.resize(frame,
                           (self.fr_size[0] // self.pixel_size, self.fr_size[1] // self.pixel_size),
                           interpolation=cv2.INTER_NEAREST)
        frame = cv2.resize(frame,
                           self.fr_size,
                           interpolation=cv2.INTER_NEAREST)
        return frame

class Pixelate_grad_filter:
    def __init__(self, pixel_size_start, pixel_size_end, time_per_res=5*10**5, pixel_size_delta=5):
        self.px_size_end = pixel_size_end
        self.px_size_start = pixel_size_start
        self.pixelate_filter = Pixelate_filter(pixel_size_start)
        self.pixel_size_delta = pixel_size_delta
        self.time_per_res = time_per_res #time spent on each grain size
        self.grow = True
        self.res_change_time = datetime.now()

    def apply_filter(self, frame):
        time_dif = (datetime.now() - self.res_change_time).microseconds
        if (time_dif >= self.time_per_res):
            if self.grow:
                self.pixelate_filter.pixel_size += self.pixel_size_delta
                if self.pixelate_filter.pixel_size > self.px_size_end:
                    self.pixelate_filter.pixel_size = self.px_size_end
                    self.px_size_end, self.px_size_start = self.px_size_start, self.px_size_end
                    self.grow = False
            else:
                self.pixelate_filter.pixel_size -= self.pixel_size_delta
                if self.pixelate_filter.pixel_size < self.px_size_end:
                    self.pixelate_filter.pixel_size = self.px_size_end
                    self.px_size_end, self.px_size_start = self.px_size_start, self.px_size_end
                    self.grow = True
            self.res_change_time = datetime.now()
        return self.pixelate_filter.apply_filter(frame)
