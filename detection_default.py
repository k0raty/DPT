
"""
Created on Thu Sep 29 09:04:03 2022

@author: anton
"""

import math
import numpy as np
import pandas as pd
from numpy.fft import fft, fftfreq
import matplotlib.pyplot as plt
import scipy.signal
import sys

class detection_default():
    """
    Fs = 50000*2;
    Duree = 10;        %Durée d'acquisition en secondes
    L = Duree*Fs;                %   longueur du signal (en échantillons de temps)
    NFFT = 2^nextpow2 (  L );                 % Next power of 2 from length of L
    t = (1:Fs*Duree)/Fs; % Echelle des temps
    """
    
    def __init__(self,is_sig= False, Fs = 50000*2, duree = 10,nombre_signal=5):
        
        self.Fs = Fs
        self.nombre_signal=nombre_signal
        self.duree = duree
        self.L = duree*Fs
        self.NFFT = self.nextpow2()
        self.t = np.arange(1,Fs*duree+1,1) / Fs
        self.signal=pd.DataFrame(columns=['secondes','signal','num'])
        self.signal_mgnet_fft=pd.DataFrame(columns=['frequence','Mag_fournier_transform','num'])
        self.signal_mgnet_fft_filtered=pd.DataFrame(columns=['frequence','Mag_fournier_transform'])

        
        ###Opening signal###
        
        if is_sig == True :
            for i in range(0,nombre_signal):
                signal = open(f'mesure01\Signal_{i}.sig','r')
                Amplitude=[]
                Temps=[]
                for x in signal :
                    sep=x.find('\t')
                    Temps.append(float(x[:sep]))
                    Amplitude.append(float(x[sep+1:-1]))
                current_df=self.plot_signal(Temps,Amplitude,columns=['secondes','signal','num'],num=i,logy=False)
                df=self.signal.append(current_df)
                self.signal=df
        else:
            for i in range(0,nombre_signal):
                
                current_df = pd.read_csv(f'Signal_{i}.csv', sep=',')
                current_df = current_df[['0','1']]
                current_df = current_df.rename(columns={"0": "secondes", "1": "signal"})
                current_df['num'] = i
                current_df.plot(x='secondes', y = 'signal')
                print(current_df)
                df=self.signal.append(current_df)
                print(df)

                self.signal=df
   

    def nextpow2(self):
        return 1 if self.L == 0 else 2**math.ceil(math.log2(self.L))
    
    def method_cepstrale(self , x0 = None, x1 = None):
        """
        Met en évidence les composantes périodiques d’un spectre - Utilisation en complément d’autres techniques
        - Permet de localiser et déterminer l’origine des défauts induisant des chocs périodiques
        - Interprétation de spectres complexes
        
        Le cepstre C est la transform´ee de Fourier appliqu´ee au logarithme de la transform
        ´ee de Fourier d’une variable x(t) (2.22). Le r´esultat s’exprime selon une variable
        uniforme au temps : les qu´efrences q. La transform´ee de Fourier d’un signal permet
        de mettre en ´evidence les p´eriodicit´es d’un signal temporel. Ainsi, le cepstre met
        en ´evidence les p´eriodicit´es d’une transform´ee de Fourier. Le cepstre fournit donc
        une information sur l’existence de peignes de raies ainsi que sur leurs fr´equences
        """
        self.filtrage(x0,x1) #Tranformée du signal moyen
        
        signal_cepstrale_wrapper = self.signal_mgnet_fft_filtered.copy()
        
        #Elevation logarithmique
        signal_cepstrale_wrapper['Mag_fournier_transform'] = signal_cepstrale_wrapper['Mag_fournier_transform'].apply(
            np.log2,
            axis=1
        )
        
        #Transformée de Fournier
        SF=self.fournier_transform(signal_cepstrale_wrapper['Mag_fournier_transform'])
        w,mag_fft= self.fournier_magnitude(SF)
        
        #Plotting
        columns=['quefrence (q)', 'Cepstre']
        
        self.signal_cepstrale=self.plot_signal(w,mag_fft,columns,logy=True)
    def pre_filtrage(self):
        
        """
        Transformée de fournier des signaux afin d'identifier les fréquences sur lesquelles travailler
        Nécessaire pour n'importe quel méthode
        """
        #1rt step :Mag_fournier_transform
        SM_list=[]
        for i in range(0,self.nombre_signal):
            signal=self.signal['signal'][self.signal['num']==i]
            S = self.fournier_transform(signal)
            SM_list.append(S)
            SM= self.mean_SM(SM_list)
            w,mag_fft=self.fournier_magnitude(SM)
            
            
            #Add to dataframe
            current_df=self.plot_signal(w,mag_fft,columns=['frequence','Mag_fournier_transform','num'],num=i)

            #Saving plot
            df=self.signal_mgnet_fft.append(current_df)
            self.signal_mgnet_fft=df
            
        print('Tapez la valeur de F1 et de F2 dans la méthode method_envelop_2')
            # Display grid
        #2nd step   
    def filtrage(self,x0,x1):
        
        """
        Transformée de fournier des signaux selon le filtrage nécessaire.
        Utile pour n'importe quelle méthode
        """
        if (x0== None) | (x1 == None): #Pas de filtrage
            return self.signal['signal']
        
        b,a=self.pass_band_filter(x0, x1)
        sf_list=[]
        
        #sf signals
        for i in range(0,self.nombre_signal):
            signal=self.signal['signal'][self.signal['num']==i]
            sf=scipy.signal.lfilter(b, a, signal)
            sf_list.append(sf)
            
        #Signal_moyen
        sfM=self.mean_SM(sf_list)
        
        
        self.df_sfM=self.plot_signal(self.t,sfM,columns=['seconde','signal_moyen'],logy=False)
        
        
        SF=self.fournier_transform(self.df_sfM['signal_moyen'])
        w,mag_fft= self.fournier_magnitude(SF)
        
        #Plotting
        columns=['frequence', 'Mag_fournier_transform']
      
        self.signal_mgnet_fft_filtered=self.plot_signal(w,mag_fft,columns,logy=True)
        
        return sfM
        
    def enveloppe(self, x0 = None ,x1 = None):
        """
        Méthode de l'enveloppe
        """
        
        sfM = self.filtrage(x0,x1)
        #Hilbert
        H=self.hilbert_transform(sfM)
        
        self.Hilbert_transform=self.plot_signal(self.t,H,columns=['seconde','amplitude'],logy=False,title = "Transformée de Hilbert")

        self.envelop_signal=abs(fft(H,self.NFFT))
        wprime=np.arange(1,self.NFFT/2+2,1)/(self.NFFT/2 +1)*(self.Fs/2)#Commence à 1 ici
        mag_fft_prime=self.envelop_signal[1:int(self.NFFT/2 + 2)]

        #Plotting signal
        self.df_envelop_signal=self.plot_signal(wprime,mag_fft_prime,columns=['fréquence (Hz)','Amplitude'],logy=False)
        
    
   
    def pass_band_filter(self,x0,x1):
        assert len(self.signal_mgnet_fft)>0,"Veuillez d'abord procéder à la première méthode !"

        print('F1 is: ',x0)
        print('F2 is: ',x1)
        
        #Fréquences extrêmes de notre filtre
        X=np.array([x0,x1])*2/self.Fs
        b,a=scipy.signal.ellip(4,0.1,30,X,'bandpass')
        w,H= self.mfreqz(b,a)
        plt.plot(w*self.Fs/(2*math.pi),abs(H));
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Mag. of frequency response')
        return b,a
        
    def fournier_transform(self,signal):
        S = 2*abs(fft(signal.to_numpy(),self.NFFT))
        return S
    
    def hilbert_transform(self,signal):
        H=abs(( scipy.signal.hilbert(signal)))
        return H
    
    def fournier_magnitude(self,SM):
        #Renvoie la magnitude du signal
        S2 = SM**2
        w=np.arange(0,self.NFFT/2+2,1)/(self.NFFT/2 +1)*(self.Fs/2)    
        mag_fft=S2[:int(self.NFFT/2 + 2)]
        return w,mag_fft
    
    def mean_SM(self,SM_list: list):
        SM=sum(SM_list)/len(SM_list)
        return SM
    
    def mfreqz(self,b, a,worN=512):
        
        # Function to depict magnitude
        # and phase plot
        # Compute frequency response of the
        # filter using signal.freqz function
        wz, hz = scipy.signal.freqz(b, a, worN=worN)
     
        # Calculate Magnitude from hz in dB
        Mag = 20*np.log10(abs(hz))
     
        # Calculate phase angle in degree from hz
        Phase = np.unwrap(np.arctan2(np.imag(hz),
                                     np.real(hz)))*(180/np.pi)
        Fs=self.Fs
        # Calculate frequency in Hz from wz
        Freq = wz*Fs/(2*np.pi)
     
        # Plot filter magnitude and phase responses using subplot.
        fig = plt.figure(figsize=(10, 6))
     
        # Plot Magnitude response
        sub1 = plt.subplot(2, 1, 1)
        sub1.plot(Freq, Mag, 'r', linewidth=2)
        sub1.axis([1, Fs/2, -100, 5])
        sub1.set_title('Magnitude Response', fontsize=20)
        sub1.set_xlabel('Frequency [Hz]', fontsize=20)
        sub1.set_ylabel('Magnitude [dB]', fontsize=20)
        sub1.grid()
     
        # Plot phase angle
        sub2 = plt.subplot(2, 1, 2)
        sub2.plot(Freq, Phase, 'g', linewidth=2)
        sub2.set_ylabel('Phase (degree)', fontsize=20)
        sub2.set_xlabel(r'Frequency (Hz)', fontsize=20)
        sub2.set_title(r'Phase response', fontsize=20)
        sub2.grid()
     
        plt.subplots_adjust(hspace=0.5)
        fig.tight_layout()
        plt.show()

        return wz,hz 
    
    def plot_signal(self, abscisse, ordonnee, columns ,num: int= None, logy=True,title :str= ""):
        """
        plotting signal
        
        """
        if( num == None):
            
            assert len(columns)==2, "Trop de colonnes !"
            d = {columns[0]: abscisse, columns[1]: ordonnee}
            df= pd.DataFrame(data=d)
            #Plotting current signal
            # Linear X axis, Logarithmic Y axis
        else:
            assert len(columns)==3, "Pas assez de colonnes !"
            d = {columns[0]: abscisse, columns[1]: ordonnee, columns[2]: num}
            df= pd.DataFrame(data=d)

            #Plotting current signal
            # Linear X axis, Logarithmic Y axis
        ax=df.plot(logy=logy, x=columns[0], y= columns[1], grid= True)
        ax.set_xlabel(columns[0])
        ax.set_ylabel(columns[1])
        ax.set_title(title)
        return df
        