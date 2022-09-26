# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 14:50:29 2022

@author: jbonho01
"""

import serial
import time
import threading
import numpy as np
from collections import deque

class ADCPoller:

    def __init__(self,port='COM10'):
        self.ser = serial.Serial()
        self.ser.port = port
        self.ser.baudrate = 115200
        self.ser.timeout = 0
        self.counter = 0
        self.thread = None
        self.buffer = deque(maxlen=1000)
        self.start_poller()
        print("Initialized")

    def start_poller(self):
        self.thread = threading.Thread(target=self.poller)
        self.thread.start()
        print("Started")

    def poller(self):
        print("Polling")
        self.running = True
        self.ser.open()
        while self.running:
            self.counter += 1
            lines = self.ser.readlines()
            for line in lines:
                data = line.decode().strip('\r\n')
                if len(data)==0 and data == '': continue
                if float(data) < 1000: continue
                self.buffer.append(float(data))
            time.sleep(0.05)
        self.ser.close()

    def stop_poller(self):
        self.running = False
        self.thread.join()
        self.thread = None
