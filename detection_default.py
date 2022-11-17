
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
from tqdm import tqdm
from scipy.stats import norm
from scipy.stats import laplace
from scipy.stats import skew
from scipy.stats import  kurtosis

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
    
    def __init__(self,is_sig= True, Fs = 100000, duree = 10,nombre_signal=5, precision = 5, seuil = 1,name: str = False):
        
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
            if(not name): 
                for i in range(0,nombre_signal):
                    signal = open(f'BE\Signal_{i}.sig','r')
                    Amplitude=[]
                    Temps=[]
                    for x in signal :
                        sep=x.find('\t')
                        Temps.append(float(x[:sep]))
                        Amplitude.append(float(x[sep+1:-1]))
                    current_df=self.plot_signal(Temps,Amplitude,columns=['secondes','signal','num'],num=i,logy=False)
                    df=self.signal.append(current_df)
                    self.signal=df
            else: #Si un nom de signal en particulier est fourni
                signal = pd.read_csv(name,sep=',',index_col=0)
                assert len(signal.columns) == 2 , "Attention , le signal entrant au format csv est temporel constitué de deux colonnes [temps,signal] uniquement."
                signal = signal.rename(columns={signal.columns[0]: "secondes", signal.columns[1]: "signal"})
                signal['num'] = 0
                self.signal = signal
                self.t = signal['secondes'].to_numpy()
                self.nombre_signal = 1
                signal.plot(x='secondes', y = 'signal')

        else:
            for i in range(0,nombre_signal):
                
                current_df = pd.read_csv(f'Signal_{i}.csv', sep=',')
                current_df = current_df[['0','1']]
                current_df = current_df.rename(columns={"0": "secondes", "1": "signal"})
                current_df['num'] = i
                current_df.plot(x='secondes', y = 'signal')
                df=self.signal.append(current_df)

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
    def filtrage(self,x0,x1,signal = pd.DataFrame()):
        
        """
        Transformée de fournier des signaux selon le filtrage nécessaire.
        Utile pour n'importe quelle méthode
        """
        
        b,a=self.pass_band_filter(x0, x1)

        if len(signal) == 0: #Pas de filtrage mais moyenne tout de même
            
        
            sf_list=[]
            
            #sf signals
            for i in range(0,self.nombre_signal):
                signal=self.signal['signal'][self.signal['num']==i]
                sf=scipy.signal.lfilter(b, a, signal)
                sf_list.append(sf)
                
            #Signal_moyen
            sfM=self.mean_SM(sf_list)
            
            self.df_sfM=self.plot_signal(self.t,sfM,columns=['seconde','signal_moyen'],logy=False)
            
            sfM = pd.DataFrame(sfM, columns = ["signal"])

        else :
            sfM = pd.DataFrame(signal['signal'], columns = ["signal"])

        
        return sfM
        
    def enveloppe(self, x0 = 7000 ,x1 = 10000 , cepstrum = True , signal = pd.DataFrame()):
        """
        Méthode de l'enveloppe
            -signal = self.icwt_signal
        """
        
        if len(signal) == 0:
            sfM = self.filtrage(x0,x1 ,signal = signal)
            
            #Cas où on analyse l'enveloppe du signal filtré par ondelette
            #On zoom sur les fréquences concernées - pas besoin d'effectuer un second filtrage.
            
            #Affichage temporel du signal
            fig,ax = plt.subplots()
            ax.plot(self.t,sfM['signal'])
            ax.set_title("Profil du signal temporel à analyser")
            
            
            SF=self.fournier_transform(sfM['signal'])
            w,mag_fft= self.fournier_magnitude(SF)
            
            
            #Plotting
            columns=['frequence', 'Mag_fournier_transform']
            #Ici on remarque bien que la transformée conserve globalement les fréquences visées sur une précision de 20Hz.
            self.signal_mgnet_fft_filtered=self.plot_signal(w,mag_fft,columns,logy=True, title = "Transformée de Founier du signal temporel")
            #Hilbert
            H=self.hilbert_transform(sfM['signal'])
            
            self.Hilbert_transform=self.plot_signal(self.t,H,columns=['seconde','signal'],logy=False,title = "Transformée de Hilbert")
            #Ondelette sur la transformée de hilbert ? -> Oui ! Voir la diff en prenant la valeur abs du signal retourné.
            self.Hilbert_transform.to_csv('matlab/signal_temporel_filtre.csv')
        
        else : 
            H = signal['signal'].to_numpy()
            #Affichage temporel du signal
            fig,ax = plt.subplots()
            ax.plot(self.t,H)
            ax.set_title("Profil du signal temporel à analyser")

        self.envelop_signal=abs(fft(H,self.NFFT))
        wprime=np.arange(1,self.NFFT/2+2,1)/(self.NFFT/2 +1)*(self.Fs/2)#Commence à 1 ici
        self.freq_vector = wprime
        mag_fft_prime=self.envelop_signal[1:int(self.NFFT/2 + 2)]
        
        #Plotting signal
        self.envelop_signal=self.plot_signal(wprime,mag_fft_prime,columns=['fréquence (Hz)','Amplitude'],title = "enveloppe du signal", logy=False)
        #Analyse Cepstrale
        if cepstrum == True :
            self.cepstre(mag_fft_prime, wprime)
        
    def cepstre(self,X,freq_vector):
        
        """
        Cepstrum of a signal.
        
        Met en évidence les composantes périodiques d’un spectre - Utilisation en complément d’autres techniques
        - Permet de localiser et déterminer l’origine des défauts induisant des chocs périodiques
        - Interprétation de spectres complexes
        - Les quéfrences caratéristiques de fonctionnement du roulement réapparâissent après ondelette.
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
        columns=['quefrence', 'cepstre']
        
        self.cepstrum=self.plot_signal(quefrency_vector,np.abs(cepstrum),columns,logy=False, title = "Cepstre du signal")

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
        cepstrum = self.cepstrum['cepstre'].to_numpy()
        quefrency_vector = self.cepstrum['quefrence'].to_numpy()
        
        fig, ax = plt.subplots()
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
        q0
        """

        # extract peak in cepstrum in valid region
        cepstrum = self.cepstrum['cepstre'].to_numpy()
        quefrency_vector = self.cepstrum['quefrence'].to_numpy()
        valid = (quefrency_vector > 1/fmax) & (quefrency_vector <= 1/fmin)
        
        self.extract_pitch(1/fmax,1/fmin)
        max_quefrency_index = np.argmax(np.abs(cepstrum)[valid])
        q0 = quefrency_vector[valid][max_quefrency_index]
        cepstre = cepstrum[valid][max_quefrency_index]
        f0 = 1/q0
        return (f0 ,cepstre)
    
    def highlight_pics(self, df:pd.DataFrame()):
        """
        Mets en évidence les fréquences caractéristiques de chaque défaut.
        """
        assert len(self.envelop_signal) >0, "Veuillez d'abord appliquer une méthode !"
        assert len(self.defauts) >0, "Veuillez d'abord détecter les défauts !"
        
        fig, ax = plt.subplots(1, 1,figsize=(20,10))
        
        df['pics'] = df.apply(lambda x : self.envelop_signal['Amplitude'].loc[self.find_neighbours(x['f0'], self.envelop_signal, 'fréquence (Hz)')],axis = 1 )
        df['f_pics'] = df.apply(lambda x : self.envelop_signal['fréquence (Hz)'].loc[self.find_neighbours(x['f0'], self.envelop_signal, 'fréquence (Hz)')],axis = 1 )
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
            neighbour = np.argmin([abs(df['fréquence (Hz)'].loc[lowerneighbour_ind]-value), abs(df['fréquence (Hz)'].loc[upperneighbour_ind]-value)])
        return neighbours[neighbour]
        
    def detect_default(self,with_ondelette = False):
        
        assert len(self.cepstrum)>0, "Veuillez d'abord calculer le cepstre !"
        
        df=self.table_defaut.copy()
        df=df.set_index('Défauts')
        df['f_under'] = df['Hz'].shift(periods = 1, fill_value = math.floor(df['Hz'].min())).round() #Décale la colonne des Hz vers le haut et arrondi par excés
        df['f_over'] = df['Hz'].shift(periods = -1, fill_value = round(df['Hz'].max())).apply(np.floor) #Décale la colonne des Hz vers le bas et arrondi par défault (sauf le dernier)
        df['fmax'] = df.apply(lambda x : min(round(x['Hz'] + self.precision), x['f_over']), axis = 1) #Récupère le fmax adapté , + petit que la fréq du défaut suivant et pas trop éloignée de la fréq actuelle également.
        df['fmin'] = df.apply(lambda x : max(round(x['Hz'] - self.precision), x['f_under'] ), axis = 1) #Récupère le fmin adapté , + grand que la fréq du défaut précédent et pas trop éloignée de la fréq actuelle également.
        df['f0_cepstre'] = df.apply(
            lambda x: self.cepstrum_f0_detection(fmin = x['fmin'],fmax = x['fmax']),
            axis=1
        )
        df['f0'],df['cepstre'] = df.apply(lambda x : x['f0_cepstre'][0],axis=1),df.apply(lambda x : x['f0_cepstre'][1],axis=1)
        df = df.drop(columns = ['f0_cepstre'])
        df['is_default'] = df.apply(lambda x : abs(x['Hz']-x['f0']) < self.seuil*min((abs(x['Hz']-x['f0'])/2*self.precision),1) ,axis = 1)
        
        df = df[df['is_default'] == True]
        limit = self.keep_outlier(self.cepstrum[self.cepstrum['quefrence']>1/df['Hz'].iloc[-1]],'cepstre') #On ne conserve que les quéfrences au dessus d'un certains seuil
        limit = limit.sort_values(by='cepstre')['cepstre'].iloc[0]
        df = df[df['cepstre']>limit]
        df = df.filter(['Hz','f0'])
        
        print("Voici les défauts supposés : \n")
        print(df)
        df.to_excel('Défauts_suppose.xlsx')
        
        defauts = df['Hz'].copy()
        if (len(defauts)==0):
            return "Il n'y a pas de défauts"
        
        for index_defaut in tqdm(range(0,len(defauts))) :
            defaut = defauts.iloc[index_defaut]
            if (with_ondelette == True):
                self.continuous_wavelet(defaut-5,defaut+5)
                self.enveloppe(signal = self.icwt_signal)
            is_defaut = self.spectral_kurtosis(defaut-0.5, defaut+0.5)
            if (is_defaut == False):
                print('Ce défaut est annulé :',defauts.index[index_defaut])
                df = df.drop(index = defauts.index[index_defaut])
        self.defauts = df

        self.highlight_pics(df)
        return df
        
    def keep_outlier(self,df, column,m=1):
        n=0
        while n<m : #On ne conserve que les véritables outliers.
             
                fig1, ax1 = plt.subplots()
                ax1.set_title("Répartition des cepstres sur le signal démodulé")
                df.boxplot(column = column , ax = ax1)
                q75,q25 = np.percentile(df[column].dropna(),[75,25])
                intr_qr = q75-q25
                max = q75+(1.5*intr_qr)
                filter = (df[column] >= max) 
                if(len(df[filter]) > 1):
                    df=df[filter]
                else :
                    break
                n+=1
        return df
            
    def evaluate_method(self,f0s=[]): #Checkpoint
        
        sample_freq = self.Fs
        
        cepstrum = self.cepstrum['cepstre'].to_numpy()
        
        
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
        
    def continuous_wavelet(self,fmin,fmax):
        #Une précision de 1 à 10 Hz ne change pas grand chose.
        import matlab.engine

        eng = matlab.engine.start_matlab() #Start using matlab to compute icwt
        fs = self.Fs

        frequencies = np.array([8.333, 63.731, 31.865, 93.663, 72.964, 3.65, 4.683])  
        
        #Start the engine
        eng.cd('matlab', nargout=1)
        fs = matlab.double([fs])
        name = 'signal_temporel_filtre.csv'
        
        fmin = matlab.double([fmin])
        fmax = matlab.double([fmax])
        icwt_signal = eng.cwt_process(fs,name,fmin,fmax)
        self.icwt_signal = np.asarray(icwt_signal)
        d = {'secondes': self.t, 'signal': self.icwt_signal[0]}
        self.icwt_signal = pd.DataFrame(d)
        self.icwt_signal.to_csv("icwt_signal.csv")
        plt.show()
        eng.quit()
        #pd.DataFrame(self.t, np.asarray(icwt_signal) , columns = ["secondes","signal"])
        
    def spectral_kurtosis(self,fmin,fmax,atol = 0.07,seuil_laplace = 10e-7,seuil_norm =50000, envelop_signal = pd.DataFrame()):
        """
        curvature of a function : https://tutorial.math.lamar.edu/classes/calciii/curvature.aspx
        """
        
        if len(envelop_signal) == 0:
            envelop_signal = self.envelop_signal
        assert len(envelop_signal) >0 , "Il faut avoir un signal enveloppe !"
        
        enveloppe = envelop_signal
        enveloppe_to_focus=enveloppe[enveloppe['fréquence (Hz)']>fmin]
        enveloppe_to_focus=enveloppe_to_focus[enveloppe_to_focus['fréquence (Hz)']<fmax]
        pic = enveloppe_to_focus['fréquence (Hz)'][enveloppe_to_focus['Amplitude'] == enveloppe_to_focus['Amplitude'].max()].iloc[0]
        del(enveloppe_to_focus)
        enveloppe=enveloppe[enveloppe['fréquence (Hz)']>pic-0.5]
        enveloppe=enveloppe[enveloppe['fréquence (Hz)']<pic+0.5]

        #Aperçu des fréquences évaluées :
        enveloppe.plot(x ='fréquence (Hz)' ,y = "Amplitude",title = "Zoom sur le spectre centré en %f"%pic, kind = 'bar')
        
        enveloppe['next_value'] = enveloppe['Amplitude'].shift(periods = -1, fill_value = 0) #Décale la colonne des Hz vers le haut et arrondi par excés
        enveloppe['previous_value'] = enveloppe['Amplitude'].shift(periods = 1, fill_value = 0) #Décale la colonne des Hz vers le haut et arrondi par excés
        
        m = []
        M = []
        max_amp = enveloppe['Amplitude'].max()
        enveloppe['is_ok'] = enveloppe.apply(lambda x: self.is_ok(M,m,pic,x['fréquence (Hz)'],x['Amplitude'],x['next_value'],x['previous_value'],max_amp),axis = 1)
        
        #On retire les pics qui empêcherai de calculer correctement le kurtosis
        enveloppe = enveloppe[enveloppe['is_ok'] == True]
        
        #On retire ceux qui seraient passé inaperçu
        current_index=enveloppe[enveloppe['fréquence (Hz)']==pic].index[0]
        current_index = list(enveloppe.index).index(current_index)
        
        #A gauche
        before_pic = enveloppe.iloc[:current_index]
        before_min = before_pic['Amplitude'].min()
        before_min_index = before_pic[before_pic['Amplitude'] == before_min].index[0]
        before_min_index = list(enveloppe.index).index(before_min_index)
        
        enveloppe = enveloppe.iloc[before_min_index:]
        
        #A droite 
        current_index=enveloppe[enveloppe['fréquence (Hz)']==pic].index[0]
        current_index = list(enveloppe.index).index(current_index)
        
        after_pic = enveloppe.iloc[current_index+1:]
        after_max = after_pic['Amplitude'].max()
        after_max_index = after_pic[after_pic['Amplitude'] == after_max].index[0]
        after_max_index = list(enveloppe.index).index(after_max_index)
        
        if after_max_index > current_index+1:
            enveloppe = enveloppe.iloc[:after_max_index]
        #On centre sur la valeur principal si il y a plus de 4 valeurs :

        #On identifie ses voisins
        if len(enveloppe) > 4:
            length_inf = len(enveloppe.index[:current_index])
            length_sup = len(enveloppe.index[current_index+1:])
            length = abs(length_sup-length_inf)
            
            if length_inf > length_sup:
                enveloppe=enveloppe.iloc[length:]
            elif length_inf < length_sup:
                enveloppe=enveloppe.iloc[:-length]
                
            assert len(enveloppe) >=3, "ce n'est pas un pic"
        
        enveloppe['mean'] = enveloppe.apply(lambda x:x['Amplitude']*x['fréquence (Hz)'],axis =1) #Le poids de chaque 

        mean = sum(enveloppe['mean'])/sum(enveloppe['Amplitude'])
        
        enveloppe['deviation'] = enveloppe.apply(lambda x: x['Amplitude']*(x['fréquence (Hz)'] - mean)**2,axis =1)
        enveloppe['deviation_scale'] = enveloppe.apply(lambda x: x['Amplitude']*abs((x['fréquence (Hz)'] - mean)),axis =1) #Le poids de chaque 

        enveloppe.plot(x ='fréquence (Hz)' ,y = "Amplitude",title = "Zoom sur le spectre arrangé en %f"%pic, kind= 'bar')

        b = (1/sum(enveloppe['Amplitude']))*sum(enveloppe['deviation_scale'])
        std = np.sqrt((1/sum(enveloppe['Amplitude']))*sum(enveloppe['deviation']))
        #Distribution laplacienne
        dist = laplace(pic,b)
        dist_norm = norm(pic,std)
        hist = enveloppe['Amplitude']
        
        
        #On crée la courbe de densité de probabilité réelle
        prob = pd.DataFrame(hist/sum(enveloppe['Amplitude']) , columns = ['Amplitude'])
        prob['fréquence (Hz)'] = enveloppe['fréquence (Hz)']
        
        #Quelques tests#
        #Est-ce une prob ?
        # assert prob.sum() == 1 , "Ce n'est pas une probabilité"
        

        prob['cdf'] = prob['Amplitude'].cumsum()
        prob['cdf_approximé'] = dist.cdf(prob['fréquence (Hz)'])
        prob['cdf_approximé_norm'] = dist_norm.cdf(prob['fréquence (Hz)'])
        
        #Affichage#
        prob.plot(x ='fréquence (Hz)' ,y = ['cdf','cdf_approximé'],title = "CDF de l'approximation normale et du signal réel")
        
        
        # We consider the distribution is normal , plotting of the normal approximation
        fig , ax = plt.subplots()
        
        enveloppe['approximation_laplace'] = enveloppe['fréquence (Hz)'].apply(lambda x: dist.pdf(x))
        enveloppe['approximation_normale'] = enveloppe['fréquence (Hz)'].apply(lambda x: dist_norm.pdf(x))

        #Remise à échelle :
        pace = 1/sum(enveloppe['approximation_laplace'])
        pace_norm = 1/sum(enveloppe['approximation_normale'])
        M= np.arange(pic-0.5,pic+0.5,0.01)
        distribution = dist.pdf(M)*pace
        distribution_norm = dist_norm.pdf(M)*pace_norm

        
        ax.plot(M,distribution,label='densité laplacienne continue')
        ax.plot(M,distribution_norm,label='densité normale continue')

        ax.plot(prob['fréquence (Hz)'],prob['Amplitude'], label = 'probabilité discrétisée')
        
        #Remise à niveau
        #Laplace
        distribution=distribution*sum(enveloppe['Amplitude'])
        ecart = distribution.max() - hist.max()  
        distribution += -ecart
        #Gauss
        distribution_norm=distribution_norm*sum(enveloppe['Amplitude'])
        ecart = distribution_norm.max() - hist.max()  
        distribution_norm += -ecart
        
        #Affichage
        fig , ax2 = plt.subplots()

        ax2.plot(M,distribution,label='approximation laplacienne continue')
        ax2.plot(M,distribution_norm,label='approximation normale continue')
        ax2.plot(prob['fréquence (Hz)'],hist, label = 'spectre discrétisée')
        ax2.legend()
        ax2.set_title("Densités de probabilité du signal")
        #Précis ? Si ça ne l'est pas , la distribution n'est pas laplacienne et est très suceptible de ne pas être un défaut.
        
        #On cherche l'index de la fréquence cummulante
        try : #Try car il peut y avoir des configuration ou cela ne fonctionne pas ( le pic est à l'extremité ect ... on considère alors que cela est faux)
            current_index=prob[prob['fréquence (Hz)']==pic].index[0]
            current_index = list(prob.index).index(current_index)
            
            #On identifie ses voisins
            index_inf = prob.index[current_index-1]
            index_sup = prob.index[current_index+1]
            prob=prob.loc[index_inf:index_sup]
            assert len(prob) >=3, "ce n'est pas un pic"
        
            #Laplace
            M= prob['fréquence (Hz)'].to_numpy()
            distribution=dist.pdf(M)*pace
            ecart = distribution.max() - prob['Amplitude'].max()  
            distribution += -ecart
            
            #Normale
            distribution_norm=dist_norm.pdf(M)*pace
            ecart = distribution_norm.max() - prob['Amplitude'].max()  
            distribution_norm += -ecart
        except : 
            print("Ce n'est pas un pic")
            return False
        
        
        if (np.allclose(distribution , prob['Amplitude'],rtol = 0 , atol = atol) == False and np.allclose(distribution_norm , prob['Amplitude'],rtol = 0 , atol = atol) == False): #Appel récursif
            print("Approximation erronée à %f "%(atol*100) +'%')
            if (len(enveloppe) == 5) : #Il peut arriver que la fréquence de juste à côté soit quasiment au même niveau que le pic , on la retire.
                thresh = enveloppe['Amplitude'].max()
                envelop_signal = enveloppe[enveloppe['Amplitude'] < thresh/2]
                envelop_signal=envelop_signal.append(enveloppe[enveloppe['Amplitude'] == thresh])
                envelop_signal=envelop_signal.sort_values(by = ['fréquence (Hz)'])
            else : envelop_signal = enveloppe.iloc[1:-1]
            if (len(envelop_signal) < 3):
                print("Ce n'est pas un pic")
                return False
            return self.spectral_kurtosis(fmin,fmax,atol,envelop_signal = envelop_signal )
        else :
            #Evaluation de la courbure / kurtosis :f''/((1+f'**2)**(3/2))
            #data = dist.rvs(size=1000000) #Big size to get the decrease random influence.
            #K = kurtosis(data, fisher=True)
            #Sk = skew(data)
            if (np.allclose(distribution , prob['Amplitude'],rtol = 0 , atol = atol) == True):
                D = sum(enveloppe['Amplitude'])*pace
                Kappa = D/(2*b**3*(1+D**2/(4*b**4))**(3/2))
                if( Kappa < seuil_laplace) :
                    print("Signal impulsif laplacien : c'est un défaut")
                    print("Kappa : ", Kappa)
                    return True
                else :
                    print("Signal non impulsif laplacien: ce n'est pas un défaut" %Kappa)
                    print("Kappa : ", Kappa)
                    return False
            else :
                D = sum(enveloppe['Amplitude'])*pace_norm
                Kappa = D/(np.sqrt(2*np.pi)*std**3)
                if( Kappa > seuil_norm) :
                    print("Signal impulsif normale: c'est un défaut")
                    print("Kappa : ", Kappa)
                    return True
                else :
                    print("Signal non impulsif normale : ce n'est pas un défaut" %Kappa)
                    print("Kappa : ", Kappa)
                    return False
        
            
    def is_ok(self,M,m,pic,current_fq,current_amp,next_amp,previous_amp,max_amp):
        
        #On conserve tout de même les fréquences voisines pour éviter un déséquilibrage dans le pas entre chaque fréquence
        if (next_amp == max_amp):
            return True
        if (previous_amp == max_amp):
            return True 
        if (current_fq< pic):
            if (current_amp < next_amp):
                if (len(M) > 0): 
                    if (current_amp > max(M)):
                        M.append(current_amp)
                        return True
                    elif (current_amp < min(M)):
                        M.append(current_amp)
                        return True
                    else : return False
                else :
                    M.append(current_amp)
                    return True
            else :
                return False
            
        elif (current_fq > pic):
            if (previous_amp > current_amp):
                if (len(m) > 0): 
                    if (current_amp < min(m)):
                        m.append(current_amp)
                        return True
                    else : return False
                else :
                    m.append(current_amp)
                    return True
            else :
                return False
        elif (current_fq == pic):
            return True
        
    def get_kurtosis(self,signal,discretisation_pace= 0.5):
        """
        - Get the kurtosis of a time signal by analysing its probability distribution
        - Comes after cwt wavelet analysis , it can tell whether a frequency in the specter corresponds to a pulse or a sinusoidal signal.
        More info on the thesis given by :
            Mohamed EL BADAOUI
            Ingénieur ISTASE
            Contribution au Diagnostic Vibratoire des Réducteurs Complexes à
            Engrenages par l’Analyse Cepstrale , 1999.
            p50
        - approximation of the distribution by a normal distribution: 
            https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.norm.html
        Parameters
        ----------
        signal : signal to analyse , dataframe with two columns ['secondes','signal'] - Continuous distribution.
        discretisation_pace : in % , correspond to the approx of the signal
        Returns
        -------
        Kurtosis - whether it is a choc (defaut) or not.
        
        La valeur K du kurtosis dépend fortement de la forme des signaux. 
        Par exemple :
            K=1.5 pour une vibration de type sinusoïdal.
            K=3 pour une vibration de type impulsionnel aléatoire.
            K élevé pour une vibration de type impulsionnel périodique.
        """
        #Discretisation of the signal

        length = signal['signal'].max() - signal['signal'].min()
        pace = length*(discretisation_pace/100)
        signal.sort_values(by = 'signal')
        signal['signal'].describe()
        
        # Get a Normal Distribution instance for our height data.
        dist = norm(signal['signal'].mean(), signal['signal'].std())
        
        signal['discrete'] = signal['signal'].apply(lambda x: (x//pace)*pace)

        # x-axis: all height data points
        # y-axis: probability for each individual data point. (calculated by pdf())
        
 
        #On crée l'histogramme à la main car la méthode hist de pandas n'est pas assez précise
        hist = signal['discrete'].value_counts()
        hist = hist.sort_index()
        
        #On crée la courbe de densité de probabilité réelle
        prob = hist/len(signal)
        
        #Quelques tests#
        #Est-ce une prob ?
        # assert prob.sum() == 1 , "Ce n'est pas une probabilité"
        
        #Précis ?
        prob = prob.reset_index()
        prob = prob.rename(columns={"index": "Amplitude"})
        prob['cdf'] = prob['discrete'].cumsum()
        prob['cdf_approximé'] = dist.cdf(prob['Amplitude'])
        assert np.allclose(prob['cdf'] , prob['cdf_approximé'],rtol = 0 , atol = 0.05) == True , "Approximation erronée à 5 %"
        
        #Affichage#
        prob.plot(x ='Amplitude' ,y = ['cdf','cdf_approximé'],title = "CDF de l'approximation normale et du signal réel")
        
        # We consider the distribution is normal , plotting of the normal approximation
        fig , ax = plt.subplots()
        signal['approximation_laplace'] = signal['signal'].apply(lambda x: dist.pdf(x))*pace
        ax.plot(signal['signal'],signal['approximation_laplace'],label='approximation normale continue')
        ax.plot(prob['Amplitude'],prob['discrete'], label = 'probabilité discrétisée')
        ax.legend()
        ax.set_title("Densités de probabilité du signal")
        
        
        #Compute kurtosis from normal approximation
        K = kurtosis(signal['approximation_laplace'])
        if( K > 3) :
            print("Signal impulsif de kurtosis %f: c'est un défaut" %K)
            return True
        else :
            print("Signal non impulsif de kurtosis %f : ce n'est pas un défaut" %K)
            return False