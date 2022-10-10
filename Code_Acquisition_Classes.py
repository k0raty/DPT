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
            
        self.task.stop
        self.task.close()
        
    def acquisition_signal(self,i):
        frequence=self.frequence
        numberOfSamples=self.numberOfSamples
        Liste = self.Liste[i]
        value = self.task.read(numberOfSamples)
        i=1
        while i<numberOfSamples :
            val = value[i]
            Liste[i,0] = i/frequence
            Liste[i,1] = val
            i=i+1
        print('SIGNAL %d OK' %i)



    def to_excel(self):
        
        for i in range(0,self.n):
            df=pd.DataFrame(self.Liste[i])
            df.to_excel('signal%d.xlsx'%i)
            print('Enregistrement %d OK' %i)
            

    def affichage(self,i):
        plt.plot(self.Liste[i][:,0],self.Liste[i][:,1])
        plt.show()


       