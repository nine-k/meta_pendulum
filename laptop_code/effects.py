import numpy as np
import cv2
import os
from datetime import datetime
from datetime import timedelta
from collections import deque
from tqdm import tqdm

#DEFAULT_SIZE = (640, 480)
#DEFAULT_SIZE = (1600, 900)
DEFAULT_SIZE = (1920, 1080)

def alpha_add(dst, dst_a, src, src_a): #alpha channels contain values from 0 to 255
    dst[:,:,:] = (dst * dst_a + src * src_a)

class No_filter:
    def __init__(self):
        pass
    def reset(self):
        pass
    def set_intensity(self, inte):
        pass
    def apply_filter(self, frame):
        pass

class Png_overlay_filter:
    def __init__(self, png_dir, fps=29,
                 dst_size=DEFAULT_SIZE,
                 seq_size=(960, 540)
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
        file_names.sort()
        print("loading pics from", png_dir)
        for frame_name in tqdm(file_names):
            tmp = cv2.imread(png_dir + frame_name, cv2.IMREAD_UNCHANGED).astype("uint8")
            tmp = cv2.resize(tmp, seq_size, interpolation=cv2.INTER_AREA)
            #overlay_mask = cv2.cvtColor(tmp, cv2.COLOR_BGR2GRAY)
            #_, overlay_mask = cv2.threshold(overlay_mask, 10, 1, cv2.THRESH_BINARY_INV)
            self.image_seq.append(tmp[:,:,0:3])
            self.image_overlay.append(tmp[:,:,3] / 255)
        self.frame_total = len(self.image_seq)
        print(self.frame_total)

    def set_intensity(self, a):
        pass
    def reset(self):
        pass



    def apply_filter(self, frame):
        cur_image = cv2.resize(self.image_seq[self.cur_frame], self.fr_size, interpolation=cv2.INTER_NEAREST)
        overlay_mask = cv2.resize(self.image_overlay[self.cur_frame], self.fr_size, interpolation=cv2.INTER_NEAREST)
        #overlay_mask = self.image_overlay[self.cur_frame]

        #overlay_mask = np.repeat(overlay_mask, 3).reshape((480, 640, 3))
        overlay_mask = overlay_mask[:,:,None]
        #frame *= (frame * (1 - overlay_mask) + cur_image * overlay_mask).astype("uint8")
        #frame += (frame * (1 - overlay_mask) + cur_image * overlay_mask).astype("uint8")
        #frame[:,:,:] = (frame * (1 - overlay_mask) + cur_image * overlay_mask).astype("uint8")
        alpha_add(frame, (1 - overlay_mask), cur_image, overlay_mask)
        time_dif = (datetime.now() - self.frame_change_time).microseconds
        #print(time_dif)
        if (time_dif >= self.frame_t_delta):
            self.cur_frame = (self.cur_frame + 1) % self.frame_total
            if (self.cur_frame == 0):
                print("done")
            self.frame_change_time = datetime.now()

class Motion_blur_filter:
    def __init__(self, memory_window=10, alpha=0.5, size=DEFAULT_SIZE):
        self.memory = deque()
        self.memory_window = memory_window
        self.memory_frame = np.zeros((size[1], size[0], 3), dtype="uint8")
        self.alpha = alpha

    def set_intensity(self, intensity):
        self.alpha = 0.0 + (1 - intensity) * 1.0

    def reset(self):
        pass

    def apply_filter(self, frame):
        self.memory.append(frame)
        #self.memory_frame += (frame / self.memory_window)
        self.memory_frame[:,:,] = cv2.addWeighted(self.memory_frame, 1,
                                                  frame, 1 / self.memory_window, 0)
        if len(self.memory) > self.memory_window:
            #self.memory_frame -= (self.memory[0] / self.memory_window)
            self.memory_frame[:,:,:] = cv2.addWeighted(self.memory_frame, 1,
                                        self.memory[0], -1/self.memory_window, 0)
            self.memory.popleft()
        frame[:,:,:] = cv2.addWeighted(self.memory_frame, (1 - self.alpha), frame, self.alpha, 0)

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
        frame += template_frame

class Displacement_zoom_filter:
    def __init__(self, template_name, offset=(20,20), frame_size=DEFAULT_SIZE, max_zoom=1):
        tmp = cv2.imread(template_name)
        tmp = cv2.resize(tmp, frame_size, interpolation=cv2.INTER_AREA)
        self.tmp = template_name

        overlay_mask = cv2.cvtColor(tmp, cv2.COLOR_BGR2GRAY)

        self.template_mask = cv2.threshold(overlay_mask, 10, 1, cv2.THRESH_BINARY_INV)[1][:,:,None]
        self.mask = cv2.threshold(overlay_mask, 10, 1, cv2.THRESH_BINARY_INV)[1][:,:,None]

        self.offset = np.array(offset)
        self.max_zoom = max_zoom
        self.fr_size = np.array(frame_size, dtype="int")

    def set_intensity(self, intensity):
        #self.mask = self.template_mask
        #return
        gain = 1 + intensity * self.max_zoom
        borders = (gain * self.fr_size - self.fr_size) / 2
        borders = borders.astype("int")
        self.mask = cv2.resize(self.template_mask, None, fx=gain, fy=gain,
                                        interpolation=cv2.INTER_AREA)[borders[1]:borders[1] + self.template_mask.shape[0],
                                                                      borders[0]:borders[0] + self.template_mask.shape[1],
                                                                      None]
    def reset(self):
        pass

    def apply_filter(self, frame):
        template_frame = frame * self.mask
        template_frame = np.roll(template_frame, self.offset, (0, 1))
        frame *= np.roll((1 - self.mask), self.offset, (0, 1))
        frame += template_frame

class Horizontal_distort_effect:
    def __init__(self, lo=0, hi=10, frame_size=DEFAULT_SIZE, MAX=50):
        self.offsets = np.random.randint(lo, hi, frame_size[1], "int8")
        self.lo = lo
        self.hi = hi
        self.MAX = MAX
        self.frame_size = frame_size

    def set_intensity(self, intensity):
        self.hi = int(self.MAX * intensity) + self.lo + 1

    def reset(self):
        pass

    def apply_filter(self, frame):
        self.offsets = np.random.randint(self.lo, self.hi, self.frame_size[1], "int8")
        for row in range(len(self.offsets)):
            frame[row, :, :] = np.roll(frame[row, :, :], self.offsets[row], 0)

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
    def __init__(self, delta=0.08, phase_delta=0.1, val_range=3, MAX_RANGE=5, frame_size=DEFAULT_SIZE):
        self.delta = delta
        self.phase = 0
        self.phase_delta = phase_delta
        self.v_range = val_range
        self.MAX_RANGE = MAX_RANGE

    def set_intensity(self, intensity):
        self.v_range = intensity * self.MAX_RANGE

    def reset(self):
        pass

    def apply_filter(self, frame):
        for col in range(frame.shape[1]):
            frame[:, col, :] = np.roll(frame[:, col, :], int(np.sin(self.phase + self.delta * col) * self.v_range), 0)
        self.phase += self.phase_delta

class Mirror_effect:
    def __init__(self, thresh=0.5):
        self.thresh = 0.5
        self.mirror = 1
    def set_intensity(self, intensity):
        if intensity > self.thresh:
            self.mirror = -1
        else:
            self.mirror = 1
    def reset(self):
        pass

    def apply_filter(self, frame):
        rows, cols, _ = frame.shape
        mid_frame = frame[:, cols // 4: 3 * cols // 4, :]
        mirrored_mid_frame = cv2.flip(mid_frame, self.mirror)
        frame[:,:,:] = np.concatenate((mid_frame, mirrored_mid_frame), axis=1)

class Pixelate_filter:
    def __init__(self, pixel_size=20, fr_size = DEFAULT_SIZE):
        self.pixel_size = pixel_size
        self.fr_size = fr_size

    def set_intensity(self, intensity):
        self.pixel_size = int(intensity * 40) + 1

    def reset(self):
        pass

    def apply_filter(self, frame):
        tmp = cv2.resize(frame,
                           (self.fr_size[0] // self.pixel_size, self.fr_size[1] // self.pixel_size),
                           interpolation=cv2.INTER_NEAREST)
        frame[:,:,:] = cv2.resize(tmp,
                           self.fr_size,
                           interpolation=cv2.INTER_NEAREST)

class Pixelate_grad_filter:
    def __init__(self, pixel_size_start=1, pixel_size_end=50, time_per_res=5*10**4, pixel_size_delta=7):
        self.px_size_end = pixel_size_end
        self.px_size_start = pixel_size_start
        self.pixelate_filter = Pixelate_filter(pixel_size_start)
        self.pixel_size_delta = pixel_size_delta
        self.time_per_res = time_per_res #time spent on each grain size
        self.grow = True
        self.res_change_time = datetime.now()

    def set_intensity(self,a):
        pass
    def reset(self):
        pass

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
        self.pixelate_filter.apply_filter(frame)

class Border_filter:
    def __init__(self, frame_size=DEFAULT_SIZE):
        self.kernel = np.array([[0, -1, 0],
                               [-1, 4, -1],
                               [0, -1, 0]])
        self.intensity = 100

    def set_intensity(self, intensity):
        self.intensity = 80 * (1 - intensity)

    def reset(self):
        pass

    def apply_filter(self, frame):
        for i in range(0, 3):
            frame[:,:,i] *= cv2.Canny(frame[:,:,i], self.intensity, self.intensity).astype("uint8") // 255

class Negative_filter:
    def __init__(self):
        pass

    def set_intensity(self, intensity):
        pass

    def reset(self):
        pass

    def apply_filter(self, frame):
        frame[:,:,:] = 255 - frame

#class Strange_offsets_filter:

class White_noise_filter:
    def __init__(self, size=DEFAULT_SIZE, frames=10, layer_weights=(0.5, 0.5, 0.5)):
        self.fr_size = size
        self.noise = np.random.normal(127, 127, (size[1], size[0], frames)).astype("uint8")
        self.layer_weights = np.array(layer_weights)
        self.number = 0

    def set_intensity(self, intensity):
        self.layer_weights = np.array((0.1, 0.1, 0.1)) + intensity * 0.8

    def reset(self):
        pass

    def apply_filter(self, frame):
        self.number += 1
        self.number %= self.noise.shape[2]
        print(self.noise[:,:,0].shape)
        print(frame[:,:,0].shape)
        #frame.astype("double") *= self.layer_weights
        for i in range(3):
            frame[:,:,i] = cv2.addWeighted(frame[:,:,i], 1 - self.layer_weights[i],
                            self.noise[:,:,self.number], self.layer_weights[i],
                            0)
            #frame[:,:,i] *= self.layer_weights[i]
            #frame[:,:,i] += self.noise[:, :, self.number] * self.inverse_wrights[i]

class Rotate_filter:
    def __init__(self, PNG_DIR, size=DEFAULT_SIZE, angle=20):
        tmp = (size[1], size[0])
        self.size = tmp
        self.shape = (cv2.imread(PNG_DIR, cv2.IMREAD_UNCHANGED)[:,:,3].astype("uint8") // 255)[:,:,None]
        self.shape_outer = 1 - self.shape
        self.angle = angle
        self.rot_matrix = cv2.getRotationMatrix2D((tmp[1]//2, tmp[0]//2),  angle, 1)

    def set_angle(self, angle):
        self.rot_matrix = cv2.getRotationMatrix2D((self.size[1]//2, self.size[0]//2),  angle, 1)
        self.angle = angle

    def apply_filter(self, frame):
        frame_comp = cv2.warpAffine(frame, self.rot_matrix, (self.size[1], self.size[0]))
        frame_comp *= self.shape
        frame *= self.shape_outer
        frame += frame_comp

class Rotate_grad_filter:
    def __init__(self, PNG_DIR, angle_delta=8, time_delta=10**3, max_angle=0):
        self.prev_change_time = datetime.now()
        self.angle_delta = angle_delta
        self.rotate_filter = Rotate_filter(PNG_DIR, angle=0)
        self.time_per_angle = time_delta
        self.MAX_ANGLE = max_angle
        self.grow=True

    def set_intensity(self, intensity):
        pass

    def reset(self):
        self.rotate_filter.set_angle(0)
        self.grow = True

    def apply_filter(self, frame):
        time_dif = (datetime.now() - self.prev_change_time)
        time_dif = time_dif.microseconds + 10**6 * time_dif.seconds
        if (time_dif >= self.time_per_angle):
            if self.rotate_filter.angle + self.angle_delta >= 360:
                self.rotate_filter.set_angle(self.angle_delta + self.rotate_filter.angle - 360)
            else:
                self.rotate_filter.set_angle(self.angle_delta + self.rotate_filter.angle)
            self.prev_change_time = datetime.now()
        #    if self.grow:
        #        if self.rotate_filter.angle + self.angle_delta >= self.MAX_ANGLE:
        #            self.grow = False
        #        self.rotate_filter.set_angle(self.angle_delta + self.rotate_filter.angle)
        #    else:
        #        if self.rotate_filter.angle - self.angle_delta <= -self.MAX_ANGLE:
        #            self.grow = True
        #        self.rotate_filter.set_angle(-self.angle_delta + self.rotate_filter.angle)
        self.rotate_filter.apply_filter(frame)

#class Circle_filter:
#    def __init__(self, size=DEFAULT_SIZE, radius=100, angle=20):
#        tmp = (size[1], size[0])
#        self.shape = tmp
#        self.circle_filter = np.zeros(tmp, dtype="uint8")
#        self.circle_filter = cv2.circle(self.circle_filter, (tmp[1] // 2, tmp[0] // 2), radius + 2, 1, thickness=-1)
#        self.circle_filter = self.circle_filter[:,:,None]
#        self.circle_filter_outer = np.ones(tmp, dtype="uint8")
#        self.circle_filter_outer = cv2.circle(self.circle_filter_outer, (tmp[1] // 2, tmp[0] // 2), radius, 1, thickness=-1)
#        self.circle_filter_outer = self.circle_filter_outer[:,:,None]
#        self.angle = angle
#        self.rot_matrix = cv2.getRotationMatrix2D((tmp[1]//2, tmp[0]//2),  angle, 1)
#
#    def apply_filter(self, frame):
#        frame_comp = frame * (1 - self.circle_filter)
#        frame_comp = cv2.warpAffine(frame_comp, self.rot_matrix, (self.shape[1], self.shape[0]))
#        frame *= self.circle_filter_outer
#        return frame_comp + frame

class Color_plane_filter:
    def __init__(self, color=(0xb4, 0x69, 0xff), alpha=1):
        self.color = np.array(color, dtype="double")
        self.alpha = 0

    def set_intensity(self, intensity):
        self.alpha = intensity

    def reset(self):
        pass

    def apply_filter(self, frame):
        #calc projection
        #frame = 255 - frame
        #proj_coeffs = np.dot(frame, self.color) / np.dot(self.color, self.color)
        #proj_coeffs = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) / 255
        proj_coeffs = (self.color * cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)[:,:,None] / 255).astype("uint8")
        frame[:,:,:] = cv2.addWeighted(proj_coeffs, self.alpha,
                                           frame, 1 - self.alpha, 0)

class Duatone_filter:
    def __init__(self, threshold=60, dua_layers=(1, 2), other_layer=(0, 1)):
        self.thresh = threshold
        self.other_layer = other_layer
        self.dua_layers = dua_layers

    def set_intensity(self, intensity):
        pass

    def reset(self):
        pass

    def apply_filter(self, frame):
        tmp = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, template_mask1 = cv2.threshold(tmp, self.thresh, 1, cv2.THRESH_BINARY_INV)
        _, template_mask2 = cv2.threshold(tmp, self.thresh, 1, cv2.THRESH_BINARY)
        #frame[:,:,self.other_layer[0]] *= self.other_layer[1]
        frame[:,:,self.dua_layers[0]] *= template_mask1
        frame[:,:,self.dua_layers[1]] *= template_mask2
        frame[:,:,:] = cv2.medianBlur(frame, 5)
        #frame[:,:,:] = cv2.GaussianBlur(frame, (6,6),0)

class RGB_shift_filter:
    def __init__(self, R_SHIFT=5, G_SHIFT=5, B_SHIFT=5, GAIN=40):
        self.R_SHIFT = R_SHIFT
        self.G_SHIFT = G_SHIFT
        self.B_SHIFT = B_SHIFT
        self.GAIN = GAIN

    def set_intensity(self, intensity):
        self.R_SHIFT = int(self.GAIN * intensity) + 1
        self.G_SHIFT = int(self.GAIN * intensity) + 1
        self.B_SHIFT = int(self.GAIN * intensity) + 1

    def reset(self):
        pass

    def apply_filter(self, frame):
        frame[:,:,0] = np.pad(frame[:,:,0], ((0, 0), (self.R_SHIFT, 0)), 'constant', constant_values=(0))[:,:-self.R_SHIFT]
        frame[:,:,1] = np.pad(frame[:,:,1], ((self.G_SHIFT, 0), (0, 0)), 'constant', constant_values=(0))[:-self.G_SHIFT,:]
        frame[:,:,2] = np.pad(frame[:,:,2], ((0, self.B_SHIFT), (0, 0)), 'constant', constant_values=(0))[self.B_SHIFT:,:]
        return frame

class Kaleidoscope_filter:
    def __init__(self, HOR=True, VERT=False, size=DEFAULT_SIZE):
        self.fr_size = size
        self.hor = HOR
        self.vert = VERT


    def apply_filter(self, frame):
        if self.hor:
            frame[self.fr_size[1] // 2:,:,:] = cv2.flip(frame, 0)[self.fr_size[1] // 2:,:,:] #HOR FLIP
        if self.vert:
            frame[:, self.fr_size[0] // 2:,:] = cv2.flip(frame, 1)[:,self.fr_size[0] // 2:,:] #VER FLIP

class Kaleidoscope8_filter:
    def __init__(self, size=DEFAULT_SIZE):
        self.fr_size = size
        self.target_side = min(size[0], size[1]) // 2
        self.diag_filter = np.ones((self.target_side, self.target_side), dtype="uint8")
        self.diag_filter = np.triu(self.diag_filter)[:,:,None]

    def apply_filter(self, frame):
        fragment = frame[:self.target_side,
                                        self.fr_size[0] // 2 - self.target_side:self.fr_size[0] // 2,
                                        :] * self.diag_filter
        comp = np.empty(fragment.shape, "uint8")
        for i in range(3):
            comp[:,:,i] = np.triu(fragment[:,:,i], 1).T
        fragment += comp
        fragment = np.concatenate((fragment, cv2.flip(fragment, 1)), 1)
        fragment = np.concatenate((fragment, cv2.flip(fragment, 0)), 0)
        u_border = (self.fr_size[1] - self.target_side * 2) // 2
        l_border = (self.fr_size[0] - self.target_side * 2) // 2
        frame[u_border:self.fr_size[1] - u_border, l_border:self.fr_size[0] - l_border,:] = fragment
        frame[:self.fr_size[1] // 2,:l_border,:] = cv2.flip(frame[self.fr_size[1] // 2:, :l_border,:], 0)
        frame[:self.fr_size[1] // 2,self.fr_size[0] - l_border:,:] = cv2.flip(frame[self.fr_size[1] // 2:,
                                                                                    self.fr_size[0] - l_border:,
                                                                                    :], 0)

class Kaleidoscope_grad_filter:
    def __init__(self, thresh_int=(0.33, 0.66, 1)):
        self.kaleids = [Kaleidoscope_filter(HOR=True, VERT=False),
                        Kaleidoscope_filter(HOR=True, VERT=True),
                        Kaleidoscope8_filter()]
        self.intensity = 0
        self.thresh = thresh_int

    def set_intensity(self, intens):
        for i in range(3):
            if (intens <= self.thresh[i]):
                self.intensity = i
                break

    def apply_filter(self, frame):
        self.kaleids[self.intensity].apply_filter(frame)

    def reset(self):
        pass

class Multiply_filter:
    def __init__(self):
        self.factor = 2

    def set_intensity(self, intensity):
        #self.factor = 2**int(intensity * 3) * 2
        self.factor = int(intensity * 2) * 2 + 1

    def reset(self):
        pass

    def apply_filter(self, frame):
        small_frame = cv2.resize(frame, None,
                                 fx=1/self.factor, fy=1/self.factor,
                                 interpolation=cv2.INTER_AREA)[:frame.shape[0] // self.factor,
                                                               :frame.shape[1] // self.factor,
                                                               :]
        tmp = small_frame
        for i in range(self.factor - 1):
            tmp = np.concatenate((tmp, small_frame), axis=1)
        small_frame = tmp
        for i in range(self.factor - 1):
            tmp = np.concatenate((tmp, small_frame), axis=0)
        frame[:,:,:] = tmp
