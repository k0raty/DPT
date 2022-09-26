# -*- coding: utf-8 -*-
"""
Created on Fri Sep 16 16:24:36 2022

@author: anton
"""
import serial
from math import sqrt
a = 739
b=-3406
c=6278
delta  = (b**2 -4*a*c)/4*a
serialPort = serial.Serial(port = "COM7", baudrate=115200,
                           bytesize=8, timeout=2, stopbits=serial.STOPBITS_ONE) # Serial port to communicate with the arduino

serialString = ""                           # Used to hold data coming over UART


def get_data():
    # Wait until there is data waiting in the serial buffer
    if(serialPort.in_waiting > 0):
        
        try :
        # Read data out of the buffer until a carraige return / new line is found
            serialString = serialPort.readline()
             
            # Print the contents of the serial data
            #print(serialString.decode('Ascii'))
            
            serialList=serialString.decode('Ascii')
            serialList=float(serialList)
            distance = - sqrt((1/a)*(serialList+(pow(b,2)-4*a*c)/(4*a))) - b/(2*a);
            return distance

        except:
            return False
        