import numpy as np
import cv2
import os

_face_cascade_dir = 

class Ball_filter:
    def __init__(self, png_dir, fps=24,
                 cascade_path="/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml"
                ):
        self.image_seq = []
        self.cur_frame = 0
        self.fps = fps
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        directory = os.fsencode(png_dir)
        for frame in os.listdir(directory):
            frame_name = os.fsdecode(frame)
            if frame_name.endswith(".png"):
                tmp = cv2.imread(png_dir + frame_name)
                tmp = cv2.resize(tmp, dst_shape, interpolation=cv2.INTER_AREA)[290: 290 + 170, 184: 184 + 109, :]
                self.image_seq.append(tmp)
        self.frame_total = len(self.image_seq)

    def detect_face(frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return faceCascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(30, 30),
                    flags=cv2.cv.CV_HAAR_SCALE_IMAGE
               )


    def apply_filter(self, frame, position=(100, 100)):
        cur_image = self.image_seq[self.cur_frame]
        cur_image = np.pad(cur_image, ((0, BSHIFT), (0, 0)), 'constant', constant_values=(0))[BSHIFT:,:] (
        overlay_mask = cv2.cvtColor(cur_image, cv2.COLOR_BGR2GRAY)
        res, overlay_mask = cv2.threshold(overlay_mask, 10, 1, cv2.THRESH_BINARY_INV)

        h, w = overlay_mask.shape
        overlay_mask = np.repeat(overlay_mask, 3).reshape((h, w, 3))
        print("mask shape", overlay_mask.shape)
        print("frame shape", frame.shape)

        frame[:h, :w,] *= overlay_mask
        frame += cur_image
        self.cur_frame = (self.cur_frame + 1) % self.frame_total
        return frame
