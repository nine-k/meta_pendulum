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
        self.frame_change_time = datetime.now()
        directory = os.fsencode(png_dir)
        file_names = []
        self.fr_size = dst_size
        for frame in os.listdir(directory):
            frame_name = os.fsdecode(frame)
            if frame_name.endswith(".png"):
                file_names.append(frame_name)
        for frame_name in sorted(file_names):
            tmp = cv2.imread(png_dir + frame_name, cv2.IMREAD_UNCHANGED)
            tmp = cv2.resize(tmp, self.fr_size, interpolation=cv2.INTER_AREA)
            #overlay_mask = cv2.cvtColor(tmp, cv2.COLOR_BGR2GRAY)
            #_, overlay_mask = cv2.threshold(overlay_mask, 10, 1, cv2.THRESH_BINARY_INV)
            self.image_seq.append(tmp[:,:,0:3].astype("uint8"))
            self.image_overlay.append(tmp[:,:,3] / 255)
        self.frame_total = len(self.image_seq)
        print(self.frame_total)



    def apply_filter(self, frame):
        cur_image = self.image_seq[self.cur_frame]
        overlay_mask = self.image_overlay[self.cur_frame]

        overlay_mask = np.repeat(overlay_mask, 3).reshape((480, 640, 3))
        #frame *= (frame * (1 - overlay_mask) + cur_image * overlay_mask).astype("uint8")
        #frame += (frame * (1 - overlay_mask) + cur_image * overlay_mask).astype("uint8")
        frame = (frame * (1 - overlay_mask) + cur_image * overlay_mask).astype("uint8")
        time_dif = (datetime.now() - self.frame_change_time).microseconds
        #print(time_dif)
        if (time_dif >= self.frame_t_delta):
            self.cur_frame = (self.cur_frame + 1) % self.frame_total
            if (self.cur_frame == 0):
                print("done")
            self.frame_change_time = datetime.now()
        return frame

class Motion_blur_filter:
    def __init__(self, memory_window=10, alpha=0.5):
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
        return (self.memory_frame * (1 - self.alpha) + frame * ( self.alpha)).astype("uint8")

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
        self.offsets = np.random.randint(lo, hi, frame_size[1], "int8")
        self.lo = lo
        self.hi = hi
        self.frame_size = frame_size

    def apply_filter(self, frame):
        self.offsets = np.random.randint(self.lo, self.hi, self.frame_size[1], "int8")
        for row in range(len(self.offsets)):
            frame[row, :, :] = np.roll(frame[row, :, :], self.offsets[row], 0)
        return frame

class Horizontal_sin_effect:
    def __init__(self, delta=0.08, phase_delta=0.1, val_range=3, frame_size=DEFAULT_SIZE):
        self.delta = delta
        self.phase = 0
        self.phase_delta = phase_delta
        self.v_range = val_range

    def apply_filter(self, frame):
        for row in range(frame.shape[0]):
            frame[row, :, :] = np.roll(frame[row, :, :], int(np.sin(self.phase + self.delta * row) * self.v_range), 0)
        self.phase += self.phase_delta
        return frame

class Vertical_sin_effect:
    def __init__(self, delta=0.08, phase_delta=0.1, val_range=3, frame_size=DEFAULT_SIZE):
        self.delta = delta
        self.phase = 0
        self.phase_delta = phase_delta
        self.v_range = val_range

    def apply_filter(self, frame):
        for col in range(frame.shape[1]):
            frame[:, col, :] = np.roll(frame[:, col, :], int(np.sin(self.phase + self.delta * col) * self.v_range), 0)
        self.phase += self.phase_delta
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

class Blur_filter:
    def __init__(self, blur_intensity=15):
        self.intensity = blur_intensity
        self.pixelate_filter = Pixelate_filter(10)
        self.pixel_size = blur_intensity
        self.fr_size = DEFAULT_SIZE

    def apply_filter(self, frame):
        frame = cv2.resize(frame,
                           (self.fr_size[0] // self.pixel_size, self.fr_size[1] // self.pixel_size),
                           interpolation=cv2.INTER_NEAREST)
        frame = cv2.resize(frame,
                           self.fr_size,
                           interpolation=cv2.INTER_CUBIC)
        frame = cv2.GaussianBlur(frame, (5,5), 0.5)
        return frame

class Border_filter:
    def __init__(self, frame_size=DEFAULT_SIZE):
        self.white_mat = np.ones((frame_size[1], frame_size[0]), dtype="uint8") * 255
        self.kernel = np.array([[0, -1, 0],
                               [-1, 4, -1],
                               [0, -1, 0]])

    def apply_filter(self, frame):
        for i in range(0, 3, 1):
            #frame[:,:,i] = cv2.Canny(frame[:,:,i], 100, 200)
            frame[:,:,i] = cv2.Canny(frame[:,:,i], 35, 35).astype("uint8") // 255 * frame[:, :, i]
            #frame[:,:,i] = self.white_mat - cv2.filter2D(frame[:,:,i], -1, self.kernel)
        return frame

class Negative_filter:
    def __init__(self):
        pass

    def apply_filter(self, frame):
        return 255 - frame

#class Strange_offsets_filter:

class Duatone_filter:
    def __init__(self, threshold=127):
        self.thresh = threshold

    def apply_filter(self, frame):
        tmp = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, template_mask = cv2.threshold(tmp, self.thresh, 1, cv2.THRESH_BINARY_INV)
        frame[:,:,1] *= template_mask
        frame[:,:,2] *= (1 - template_mask)
        return frame

class Duatone_grad_filter:
    def __init__(self, start_thresh, end_thresh, delta=20, nsecs_per_thresh=1*10**4):
        self.thresh_end = end_thresh
        self.thresh_start = start_thresh
        self.duatone_filter = Duatone_filter(threshold=start_thresh)
        self.thresh_delta = delta
        self.time_per_thresh = nsecs_per_thresh #time spent on each grain size
        self.grow = True
        self.res_change_time = datetime.now()

    def apply_filter(self, frame):
        time_dif = (datetime.now() - self.res_change_time).microseconds
        if (time_dif >= self.time_per_thresh):
            if self.grow:
                self.duatone_filter.thresh += self.thresh_delta
                if self.duatone_filter.thresh > self.thresh_end:
                    self.duatone_filter.thresh = self.thresh_end
                    self.thresh_end, self.thresh_start = self.thresh_start, self.thresh_end
                    self.grow = False
            else:
                self.duatone_filter.thresh -= self.thresh_delta
                if self.duatone_filter.thresh < self.thresh_end:
                    self.duatone_filter.thresh = self.thresh_end
                    self.thresh_end, self.thresh_start = self.thresh_start, self.thresh_end
                    self.grow = True
            self.res_change_time = datetime.now()
        return self.duatone_filter.apply_filter(frame)


class White_noise_filter:
    def __init__(self, size=DEFAULT_SIZE, frames=40, layer_weights=(0.5, 0.5, 0.5)):
        self.fr_size = size
        self.noise = np.random.normal(127, 127, (size[1], size[0], frames)).astype("uint8")
        self.layer_weights = layer_weights
        self.number = 0

    def apply_filter(self, frame):
        self.number += 1
        self.number %= self.noise.shape[2]
        print(self.noise[:,:,0].shape)
        print(frame[:,:,0].shape)
        for i in range(3):
            frame[:,:,i] = (self.noise[:, :, self.number] * (1 - self.layer_weights[i]) +
                            frame[:,:,i] * self.layer_weights[i]).astype("uint8")
        return frame
