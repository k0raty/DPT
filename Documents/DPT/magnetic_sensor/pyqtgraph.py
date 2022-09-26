# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 14:36:48 2022

@author: anton
"""

#from pyqtgraph.Qt import QtGui, QtCore -> importer à la main.
#import pyqtgraph as pg
import numpy as np
import time
from magnetic_sensor.magnetic_sensor import get_data



def update(data):
    
    print("Distance is " , data)
   
class Thread(pg.QtCore.QThread):
    
    app = QtGui.QApplication([])

    
    newData = pg.QtCore.Signal(object)
    
    win = pg.GraphicsLayoutWidget(show=True, title="Basic plotting examples")

    distance = 1000*[0]
    
    def graph(self):
        
        
        
        self.win.resize(1000,600)
        self.win.setWindowTitle('pyqtgraph example: Plotting')

        lbl = self.win.addLabel("<span style='font-size: 60pt'>label</span>")

        plot = self.win.addPlot(title="courbe")
        x = np.linspace(0, 50,1000)

        y=self.distance        
        
        curve = plot.plot(pen='y')
        curve.setData(y)

        proxy = QtGui.QGraphicsProxyWidget()
        button = QtGui.QPushButton('Coucou')
        proxy.setWidget(button)

       
        self.win.addItem(proxy)#,row=1,col=1)
        button.clicked.connect(self.coucou_function)
    
    def coucou_function():
        print("Hello World!")
    
    def stop_poller(self):
        self.running = False
        self.thread.join()
        self.thread = None
        
    def run(self):
        self.graph()
        data = get_data()
        if(data): #If data does exist
            self.newData.emit(data)
            self.distance.pop(0)
            self.distance.append(data)
            timer = QtCore.QTimer()
            timer.timeout.connect(update)
            timer.start(50)
            
            

            # do NOT plot data from here!
        
##Execution is here ##
          
thread = Thread()
thread.run()
 ## Start Qt event loop unless running in interactive mode or using pyside.
if __name__ == '__main__':
     import sys
     if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
         QtGui.QApplication.instance().exec_()
         thread.stop_poller()


