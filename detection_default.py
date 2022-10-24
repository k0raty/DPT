
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
import matplotlib.collections as collections

#Suppression warning 
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
class detection_default():
    """
    Fs = 50000*2;
    Duree = 10;        %Durée d'acquisition en secondes
    L = Duree*Fs;                %   longueur du signal (en échantillons de temps)
    NFFT = 2^nextpow2 (  L );                 % Next power of 2 from length of L
    t = (1:Fs*Duree)/Fs; % Echelle des temps
    """
    
    def __init__(self,is_sig= True, Fs = 50000*2, duree = 10,nombre_signal=5, precision = 5, seuil = 0.5):
        
        self.Fs = Fs #Fréquence d'échantillonage -> dépend de Fmaxi
        self.precision = precision #Précision de l'échantillonage du cepstre
        self.seuil = seuil #|f_mesuré - f_anomalie| < Epsilon, seuil d'acceptation d'égalité entre fréquence de défaut et fréquence du cepstre.  
        self.table_defaut = pd.read_excel("Data_defauts.xlsx").sort_values(by='Hz')
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
        
    
    def pre_filtrage(self , cut = True):
        
        """
        Transformée de fournier des signaux afin d'identifier les fréquences sur lesquelles travailler
        Nécessaire pour n'importe quel méthode
        cut -> si on effectue déjà une préselection des fréquences
        """
        #1rt step :Mag_fournier_transform
        SM_list=[]
        for i in range(0,self.nombre_signal):
            signal=self.signal['signal'][self.signal['num']==i]
            S = self.fournier_transform(signal)
            SM_list.append(S)
            SM= self.mean_SM(SM_list)
            w,mag_fft=self.fournier_magnitude(SM,cut)
            
            
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
        if (x0== None) | (x1 == None): #Pas de filtrage mais moyenne tout de même
            return self.signal_mgnet_fft
        
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
        
    def enveloppe(self, x0 = None ,x1 = None , cepstrum = True):
        """
        Méthode de l'enveloppe
        """
        
        sfM = self.filtrage(x0,x1)
        #Hilbert
        H=self.hilbert_transform(sfM)
        
        self.Hilbert_transform=self.plot_signal(self.t,H,columns=['seconde','amplitude'],logy=False,title = "Transformée de Hilbert")

        self.envelop_signal=abs(fft(H,self.NFFT))
        wprime=np.arange(1,self.NFFT/2+2,1)/(self.NFFT/2 +1)*(self.Fs/2)#Commence à 1 ici
        self.freq_vector = wprime
        mag_fft_prime=self.envelop_signal[1:int(self.NFFT/2 + 2)]
        
        #Plotting signal
        self.envelop_signal=self.plot_signal(wprime,mag_fft_prime,columns=['fréquence (Hz)','Amplitude'],logy=False)
        #Analyse Cepstrale
        if cepstrum == True :
            self.cepstrum(mag_fft_prime, wprime)
        
    def cepstrum(self,X,freq_vector):
        
        """
        Cepstrum of a signal.
        
        Met en évidence les composantes périodiques d’un spectre - Utilisation en complément d’autres techniques
        - Permet de localiser et déterminer l’origine des défauts induisant des chocs périodiques
        - Interprétation de spectres complexes
        
        Le cepstre C est la transform´ee de Fourier appliqu´ee au logarithme de la transform
        ´ee de Fourier d’une variable x(t) (2.22). Le r´esultat s’exprime selon une variable
        uniforme au temps : les qu´efrences q. La transform´ee de Fourier d’un signal permet
        de mettre en ´evidence les p´eriodicit´es d’un signal temporel. Ainsi, le cepstre met
        en ´evidence les p´eriodicit´es d’une transform´ee de Fourier. Le cepstre fournit donc
        une information sur l’existence de peignes de raies ainsi que sur leurs fr´equences
        
        C'est une méthode complémentaire d'analyse.

        Returns
        -------
        Cesptrum of the signal.

        """
        log_X = np.log(np.abs(X))
        cepstrum = np.fft.rfft(log_X)
        df = freq_vector[1] - freq_vector[0]
        quefrency_vector = np.fft.rfftfreq(log_X.size, df)
        
        #Plotting
        columns=['quefrence (q)', 'Cepstre']
        
        self.cepstrum=self.plot_signal(quefrency_vector,np.abs(cepstrum),columns,logy=False)

    def extract_pitch(self,quefrence_min,quefrence_max):
        """
        One thing that makes a robust identificiation difficult is that there are also several other peaks, 
        as well as a strong component at zero quefrency. This is mentioned in the 1964 paper by Noll,
        “Short‐Time Spectrum and ‘Cepstrum’ Techniques for Vocal‐Pitch Detection.”, 
        who attenuates the low quefrency components, which are expected to be high 
        since the log magnitude of the spectrum has a nonzero mean.

        Returns
        -------
        None.

        """
        cepstrum = self.cepstrum['Cepstre'].to_numpy()
        quefrency_vector = self.cepstrum['quefrence (q)'].to_numpy()
        
        fig, ax = plt.subplots()
#        ax.vlines(0.0310, 0, np.max(np.abs(cepstrum)), alpha=.2, lw=3, label='expected peak')
        ax.plot(quefrency_vector, np.abs(cepstrum))
        valid = (quefrency_vector > quefrence_min) & (quefrency_vector <= quefrence_max)
        collection = collections.BrokenBarHCollection.span_where(
            quefrency_vector, ymin=0, ymax=np.abs(cepstrum).max(), where=valid, facecolor='green', alpha=0.5, label='valid pitches')
        ax.add_collection(collection)
        ax.set_xlabel('quefrency (s)')
        ax.set_title('cepstrum')
        ax.legend()
        
    def cepstrum_f0_detection(self, fmin=10, fmax=80):
        """
        Returns f0 based on cepstral processing.
        Permet alors de reconnaître les pics caratéristiques.
        Parameters
        ----------
        fmin : TYPE, optional
            DESCRIPTION. The default is 10.
        fmax : TYPE, optional
            DESCRIPTION. The default is 80.

        Returns
        -------
        f0 : TYPE
            DESCRIPTION.

        """
        print(fmin,fmax)
        # extract peak in cepstrum in valid region
        cepstrum = self.cepstrum['Cepstre'].to_numpy()
        quefrency_vector = self.cepstrum['quefrence (q)'].to_numpy()
        valid = (quefrency_vector > 1/fmax) & (quefrency_vector <= 1/fmin)
        
        self.extract_pitch(1/fmax,1/fmin)
        max_quefrency_index = np.argmax(np.abs(cepstrum)[valid])
        f0 = 1/quefrency_vector[valid][max_quefrency_index]
        return f0
    
    def highlight_pics(self, df:pd.DataFrame()):
        """
        Mets en évidence les fréquences caractéristiques de chaque défaut.
        """
        assert len(self.envelop_signal) >0, "Veuillez d'abord appliquer une méthode !"
        assert len(self.defauts) >0, "Veuillez d'abord détecter les défauts !"
        
        fig, ax = plt.subplots(1, 1,figsize=(20,10))
        
        df['pics']=df.apply(lambda x : self.envelop_signal['Amplitude'].loc[self.find_neighbours(x['f0'], self.envelop_signal, 'fréquence (Hz)')],axis = 1 )
        
        self.envelop_signal.plot(x= 'fréquence (Hz)', y = 'Amplitude' , ax = ax, label = "Spectre démodulé")
        
        df.plot(x = 'f0' , y = 'pics', kind = 'scatter', marker='X', ax = ax ,c = 'red', label ="Défauts détectés")
    
    def find_neighbours(self,value, df, colname):
        """
        find the nearest neighbours in a column of the dataframe 

        Parameters
        ----------
        value : value to approach
        df : dataframe
        colname : column to search in

        Returns
        -------
        Index of nearest neighbour

        """
        exactmatch = df[df[colname] == value]
        if not exactmatch.empty:
            return exactmatch.index
        else:
            lowerneighbour_ind = df[df[colname] < value][colname].idxmax()
            upperneighbour_ind = df[df[colname] > value][colname].idxmin()
            neighbours =[lowerneighbour_ind,upperneighbour_ind]
            neighbour = np.argmin([abs(df.index[lowerneighbour_ind]-value), abs(df.index[upperneighbour_ind]-value)])
        return neighbours[neighbour]
        
    def detect_default(self):
        
        assert len(self.cepstrum)>0, "Veuillez d'abord calculer le cepstre !"
        
        df=self.table_defaut.copy()
        df=df.set_index('Défauts')
        df['f_under'] = df['Hz'].shift(periods = 1, fill_value = math.floor(df['Hz'].min())).round() #Décale la colonne des Hz vers le haut et arrondi par excés
        df['f_over'] = df['Hz'].shift(periods = -1, fill_value = round(df['Hz'].max())).apply(np.floor) #Décale la colonne des Hz vers le bas et arrondi par défault (sauf le dernier)
        df['fmax'] = df.apply(lambda x : min(round(x['Hz'] + self.precision), x['f_over']), axis = 1) #Récupère le fmax adapté , + petit que la fréq du défaut suivant et pas trop éloignée de la fréq actuelle également.
        df['fmin'] = df.apply(lambda x : max(round(x['Hz'] - self.precision), x['f_under'] ), axis = 1) #Récupère le fmin adapté , + grand que la fréq du défaut précédent et pas trop éloignée de la fréq actuelle également.
        df['f0'] = df.apply(
            lambda x: self.cepstrum_f0_detection(fmin = x['fmin'],fmax = x['fmax']),
            axis=1
        )
        df['is_default'] = df.apply(lambda x : abs(x['Hz']-x['f0']) < self.seuil ,axis = 1)
        self.defauts = df
        df = df[df['is_default'] == True]
        df = df.filter(['Hz','f0'])
        df.to_excel('Défauts.xlsx')
        self.highlight_pics(df)
        return df
        
    def evaluate_method(self,f0s=[]): #Checkpoint
        
        sample_freq = self.Fs
        
        cepstrum = self.cepstrum['Cepstre'].to_numpy()
        
        
        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 5))
        
        for ax, sample_freq_var in zip([ax1, ax2], 
                                       [sample_freq, 2 * sample_freq]):
            
            f0s = np.linspace(83, 639, num=100)
            cepstrum_f0s = []
            for f0 in f0s:
                cepstrum_f0s.append(self.cepstrum_f0_detection(cepstrum, sample_freq_var))
            ax.plot(f0s, cepstrum_f0s, '.')
           # ax.plot(f0s, f0s, label='expected', alpha=.2)
            ax.legend()
            ax.set_xlabel('true f0 (Hz)')
            ax.set_ylabel('cepstrum based f0 (Hz)')
            ax.set_title(f'sampling frequency {sample_freq_var} Hz')
        
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
    
    def fournier_magnitude(self,SM, cut = True):
        #Renvoie la magnitude du signal
        S2 = SM**2
        
        if(cut) :
            w=np.arange(0,self.NFFT/2+2,1)/(self.NFFT/2 +1)*(self.Fs/2)    
            mag_fft=S2[:int(self.NFFT/2 + 2)]
        else :
            w = np.arange(0,len(S2))/(self.NFFT/2 +1)*(self.Fs/2)  
            mag_fft = S2
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
        