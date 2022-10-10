import nidaqmx
import matplotlib.pyplot as plt
from nidaqmx.constants import TerminalConfiguration
import numpy as np
import pandas as pd

class acquisition_signal_capteur() :
    """
    frequence = 100000
    T = 10 #s
    numberOfSamples = T*frequence
    Liste = np.zeros((numberOfSamples,2)) * np.nan
    Liste2 = np.zeros((numberOfSamples,2)) * np.nan
    Liste3 = np.zeros((numberOfSamples,2)) * np.nan
    Liste4 = np.zeros((numberOfSamples,2)) * np.nan
    Liste5 = np.zeros((numberOfSamples,2)) * np.nan
    """


    def __init__(self, frequence = 100000, T = 10):

        self.frequence = frequence
        self.T = T
        self.numberOfSamples = T*frequence
        self.Liste = np.zeros((numberOfSamples,2)) * np.nan
        self.Liste2 = np.zeros((numberOfSamples,2)) * np.nan
        self.Liste3 = np.zeros((numberOfSamples,2)) * np.nan
        self.Liste4 = np.zeros((numberOfSamples,2)) * np.nan
        self.Liste5 = np.zeros((numberOfSamples,2)) * np.nan


        # Démarrage Acquisition #

        task = nidaqmx.Task()
        task.ai_channels.add_ai_voltage_chan("Dev1/ai0",terminal_config=TerminalConfiguration.RSE)
        task.timing.cfg_samp_clk_timing((frequence), source='', active_edge=nidaqmx.constants.Edge.RISING, sample_mode=nidaqmx.constants.AcquisitionType.FINITE, samps_per_chan=numberOfSamples*5)
        task.start()

    def acquisition_signal1(numberOfSamples, frequence, self):
        Liste = self.Liste
        value = task.read(numberOfSamples)
        i=1
        while i<numberOfSamples :
            val = value[i]
            self.Liste[i,0] = i/frequence
            self.Liste[i,1] = val
            i=i+1
        print('SIGNAL 1 OK')

    def acquisition_signal2(numberOfSamples, frequence, self):
        value = task.read(numberOfSamples)
        i=1
        while i<numberOfSamples :
            val = value[i]
            Liste2[i,0] = i/frequence
            Liste2[i,1] = val
            i=i+1
        print('SIGNAL 2 OK')

    def acquisition_signal3(numberOfSamples, frequence, Liste3):
        value = task.read(numberOfSamples)
        i=1
        while i<numberOfSamples :
            val = value[i]
            Liste3[i,0] = i/frequence
            Liste3[i,1] = val
            i=i+1
        print('SIGNAL 3 OK')

    def acquisition_signal4(numberOfSamples, frequence, Liste4):
        value = task.read(numberOfSamples)
        i=1
        while i<numberOfSamples :
            val = value[i]
            Liste4[i,0] = i/frequence
            Liste4[i,1] = val
            i=i+1
        print('SIGNAL 4 OK')

    def acquisition_signal5(numberOfSamples, frequence, Liste5):
        value = task.read(numberOfSamples)
        i=1
        while i<numberOfSamples :
            val = value[i]
            Liste5[i,0] = i/frequence
            Liste5[i,1] = val
            i=i+1
        print('SIGNAL 5 OK')

    def to_excel(Liste, Liste2, Liste3, Liste4, Liste5):

        df1=pd.DataFrame(Liste)
        df1.to_excel('signal0.xlsx')
        print('Enregistrement 1 OK')
        df2=pd.DataFrame(Liste2)
        df2.to_excel('signal1.xlsx')
        print('Enregistrement 2 OK')
        df3=pd.DataFrame(Liste3)
        df3.to_excel('signal3.xlsx')
        print('Enregistrement 3 OK')
        df4=pd.DataFrame(Liste4)
        df4.to_excel('signal4.xlsx')
        print('Enregistrement 4 OK')
        df5=pd.DataFrame(Liste5)
        df5.to_excel('signal5.xlsx')
        print('Enregistrement 5 OK')

    def affichage(Liste, Liste2, Liste3, Liste4, Liste5):
        plt.plot(Liste[:,0],Liste[:,1],Liste2[:,0],Liste2[:,1],Liste3[:,0],Liste3[:,1],Liste4[:,0],Liste4[:,1],Liste5[:,0],Liste5[:,1])
        plt.show()


    def __end__():
        task.stop
        task.close()