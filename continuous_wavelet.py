# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 19:09:08 2022

@author: anton
"""

import numpy as np

import pywt
import matplotlib.pyplot as plt
import pandas as pd
import matlab.engine

eng = matlab.engine.start_matlab()


fs = 50000*2

dt = 1/fs

frequencies = np.array([8.333, 63.731, 31.865, 93.663, 72.964, 3.65, 4.683])  

eng.cd('matlab', nargout=1)
fs = matlab.double([100000])
name = 'signal_temporel_filtre.csv'
fmin = matlab.double([25])
fmax = matlab.double([32])
icwt_signal = eng.cwt_process(fs,name,fmin,fmax)
icwt_signal = np.asarray(icwt_signal)
eng.quit()