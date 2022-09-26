# -*- coding: utf-8 -*-
"""
Created on Mon Aug 29 12:20:46 2022

@author: jbonho01
"""

import serial
import time
import threading
import numpy as np
from collections import deque

class ADCPoller:

    def __init__(self,port='COM7'):
        self.ser = serial.Serial()
        self.ser.port = port
        self.ser.baudrate = 115200
        self.ser.timeout = 0
        self.counter = 0
        self.thread = None
        self.buffer = deque(maxlen=1000)
        self.start_poller()
        print("Initialized")

    def start_poller(self):
        self.thread = threading.Thread(target=self.poller)
        self.thread.start()
        print("Started")

    def poller(self):
        print("Polling")
        self.running = True
        self.ser.open()
        while self.running:
            self.counter += 1
            lines = self.ser.readlines()
            for line in lines:
                data = line.decode().strip('\r\n')
                if len(data)==0 and data == '': continue
                if float(data) < 1000: continue
                self.buffer.append(float(data))
            time.sleep(0.05)
        self.ser.close()

    def stop_poller(self):
        self.running = False
        self.thread.join()
        self.thread = None

if __name__ == "__main__":
    adc = ADCPoller()

        # -*- coding: utf-8 -*-
    """
    This example demonstrates many of the 2D plotting capabilities
    in pyqtgraph. All of the plots may be panned/scaled by dragging with 
    the left/right mouse buttons. Right click on any plot to show a context menu.
    """
#    from pyqtgraph.Qt import QtGui, QtCore -> importer à la main.
#    import pyqtgraph as pg

    #QtGui.QApplication.setGraphicsSystem('raster')
    app = QtGui.QApplication([])
    #mw = QtGui.QMainWindow()
    #mw.resize(800,800)

    win = pg.GraphicsLayoutWidget(show=True, title="Basic plotting examples")
    win.resize(1000,600)
    win.setWindowTitle('pyqtgraph example: Plotting')

    # Enable antialiasing for prettier plots
    pg.setConfigOptions(antialias=True)

    lbl = win.addLabel("<span style='font-size: 60pt'>0.0000 mV</span>")
    win.nextRow()

    p6 = win.addPlot(title="Updating plot")
    curve = p6.plot(pen='y')

    def update():
        global curve, data, ptr, p6
        try:
            curve.setData(adc.buffer)
            lbl.setText(f"<span style='font-size: 60pt'>{adc.buffer[-1]:.0f} mV</span>")
        except:
            print("Chargement du buffer")

    timer = QtCore.QTimer()
    timer.timeout.connect(update)
    timer.start(50)

    ## Start Qt event loop unless running in interactive mode or using pyside.
    if __name__ == '__main__':
        import sys
        if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
            QtGui.QApplication.instance().exec_()
            adc.stop_poller()