import nidaqmx
import matplotlib.pyplot as plt
from nidaqmx.constants import TerminalConfiguration
import numpy as np
import pandas as pd
import sys
class acquisition_signal_capteur() :
    """
    frequence =
    T =  #s
    numberOfSamples = T*frequence
    Liste = 
    """


    def __init__(self, frequence = 100000, T = 10, n = 5):

        self.frequence = frequence
        self.T = T
        self.n = n
        self.numberOfSamples = T*frequence
        self.Liste= [np.zeros((self.numberOfSamples,2)) * np.nan for i in range(0,n)]


        # Démarrage Acquisition #

        self.task = nidaqmx.Task()
        self.task.ai_channels.add_ai_voltage_chan("Dev1/ai0",terminal_config=TerminalConfiguration.RSE)
        self.task.timing.cfg_samp_clk_timing((frequence), source='', active_edge=nidaqmx.constants.Edge.RISING, sample_mode=nidaqmx.constants.AcquisitionType.FINITE, samps_per_chan=self.numberOfSamples*5)
        self.task.start()

        #Acquisition
        for i in range(0,n):
            self.acquisition_signal(i)

        self.to_csv()

        self.task.stop
        self.task.close()

    def acquisition_signal(self,i):
        frequence=self.frequence
        numberOfSamples=self.numberOfSamples
        Liste = self.Liste[i]
        value = self.task.read(numberOfSamples)
        k=1
        while k<numberOfSamples :
            val = value[k]
            Liste[k,0] = k/frequence
            Liste[k,1] = val
            k=k+1
        print('SIGNAL %d OK' %i)



    def to_csv(self):

        for k in range(0,self.n):
            self.Liste[k][0] = np.array([0,0]) #valeur initiale
            df=pd.DataFrame(self.Liste[k])
            df.to_csv('Signal_%d.csv'%k)
            print('Enregistrement %d OK' %k)


    def affichage(self,i):
        plt.plot(self.Liste[i][:,0],self.Liste[i][:,1])
        plt.show()
