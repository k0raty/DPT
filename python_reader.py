# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 09:49:59 2022

@author: anton
"""

import serial
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import time
#bibli grideye

size = 8
T_min = 20
T_max = 40
C_min =240
C_max =360
serialPort = serial.Serial(port = "COM7", baudrate=9600,
                           bytesize=8, timeout=2, stopbits=serial.STOPBITS_ONE) # Serial port to communicate with the arduino

serialString = ""                           # Used to hold data coming over UART

def remap( x, oMin=T_min, oMax=T_max, nMin=C_min, nMax=C_max ): #Remap the values to get better visualisation of temperature

    #range check
    if oMin == oMax:
        print ("Warning: Zero input range")
        return None

    if nMin == nMax:
        print ("Warning: Zero output range")
        return None

    #check reversed input range
    reverseInput = False
    oldMin = min( oMin, oMax )
    oldMax = max( oMin, oMax )
    if not oldMin == oMin:
        reverseInput = True

    #check reversed output range
    reverseOutput = False   
    newMin = min( nMin, nMax )
    newMax = max( nMin, nMax )
    if not newMin == nMin :
        reverseOutput = True

    portion = (x-oldMin)*(newMax-newMin)/(oldMax-oldMin)
    if reverseInput:
        portion = (oldMax-x)*(newMax-newMin)/(oldMax-oldMin)

    result = portion + newMin
    if reverseOutput:
        result = newMax - portion

    return result

while(1): #Infinit loop
    time.sleep(2) # To avoid overload.
    # Wait until there is data waiting in the serial buffer
    if(serialPort.in_waiting > 0):
        
        try :
        # Read data out of the buffer until a carraige return / new line is found
            serialString = serialPort.readline()
             
            # Print the contents of the serial data
            print(serialString.decode('Ascii'))
            
            serialList=serialString.decode('Ascii')
            
            L = []
            k = 0
            for value in serialList :
                if (value == ",") :
                    L.append(float(serialList[k:k+5]))
                    k+=6
            serialMatrix = np.zeros((size,size))
            k = 0
            
      #      L = list(map(remap,L))
    
            for i in range(size) :
                for j in range(size) :
                    serialMatrix[i][j] = L[k]
                    k+=1
            
            fig, ax= plt.subplots(1, 1)
            
            #Building the heatmap
  #          heatmap = sns.heatmap(serialMatrix, ax=ax)
        
            ax.imshow(serialMatrix, interpolation = 'bicubic', cmap='jet', vmin = 15, vmax = 40)
            plt.show()
           
            
            # Tell the device connected over the serial port that we recevied the data!
            # The b at the beginning is used to indicate bytes!
        except :
            pass
