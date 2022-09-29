# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 09:04:03 2022

@author: anton
"""

import math
import numpy as np
import pandas as pd


class detection_default() :
    """
    Fs = 50000*2;
    Duree = 10;        %Durée d'acquisition en secondes
    L = Duree*Fs;                %   longueur du signal (en échantillons de temps)
    NFFT = 2^nextpow2 (  L );                 % Next power of 2 from length of L
    t = (1:Fs*Duree)/Fs; % Echelle des temps
    """
    
    def __init__(self, Fs = 50000*2, duree = 10,nombre_signal=5):
    
        self.Fs = Fs
        self.duree = duree
        self.L = duree*Fs
        print(self.L)
        self.NFFT = 2**(self.nextpow2())
        self.t = np.arange(1,Fs*duree+1,1) / Fs
        self.signal=pd.DataFrame(columns=['s','signal','num'])
        ###Opening signal###
        for i in range(0,nombre_signal):
            print(i)
            signal = open(f'mesure01\Signal_{i}.sig','r')
            Amplitude=[]
            Temps=[]
            for x in signal :
                sep=x.find('\t')
                Amplitude.append(float(x[:sep]))
                Temps.append(float(x[sep+1:-1]))
            d = {'s': Temps, 'signal': Amplitude, 'num': i}
            df=self.signal.append(pd.DataFrame(data=d))
            self.signal=df
            self.signal[self.signal['num']==i].plot(x='signal',y='s')
    def nextpow2(self):
        return 1 if self.L == 0 else 2**math.ceil(math.log2(self.L))