import serial

import numpy as np

import time
from scipy.signal import medfilt
from collections import deque

class Sensor_d:
    def __init__(self, delta=200):
        self.values = np.array([0] * 6)
        self.sensor_values_raw = np.array([0] * 6)
        self.delta = delta
        self.grow = True

    def parse_and_filter(self):
        time.sleep(0.01)
        if self.grow:
            self.sensor_values_raw += self.delta
        else:
            self.sensor_values_raw -= self.delta
        if (self.sensor_values_raw[0] >= 9000):
            self.grow = False
        if (self.sensor_values_raw[0] <= -9000):
            self.grow = True
        self.values[3] = self.sensor_values_raw[3]
        if (abs(self.sensor_values_raw[3]) < 500):
            self.values[3] = 0


class Sensor:
    def __init__(
                 self,
                 filter_window=11,
                 offsets=(0, 0, 0, -2000, 0, 0),
                 NAME="/dev/ttyACM1",
                 ZERO_THRESH = (300, 500) #gyro, axel
                ):
        self.filter_window = filter_window
        self.value_window = [deque() for i in range(6)] #gx gy gz ax ay az
        self.values = [0] * 6
        for i in range(6):
            for _ in range(self.filter_window):
                self.value_window[i].append(0)
        self.offsets = offsets
        self.ser = serial.Serial(NAME, 38400, timeout=1.5)
        self.ZERO_THRESH = ZERO_THRESH

    def parse_and_filter(self):
        zero_count = 0
        while zero_count < 3:
            cur_byte = self.ser.read(1)[0]
            if cur_byte == 0:
                zero_count += 1
            else:
                zero_count = 0
        for i in range(6):
            self.ser.read(1) #separating byte 0xFF
            data = self.ser.read(2)
            self.value_window[i].append(int.from_bytes(data, byteorder='big', signed = True) - self.offsets[i])
            self.value_window[i].popleft()
        for i in range(6):
            self.values[i] = medfilt(self.value_window[i], self.filter_window)[self.filter_window // 2]
            if (abs(self.values[i]) <= self.ZERO_THRESH[i // 3]):
                self.values[i] = 0

    def __exit__(self, exc_type, exc_value, traceback):
        self.ser.close()
