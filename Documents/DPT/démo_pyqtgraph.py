# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 11:34:09 2022

@author: jbonho01
"""

from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg
import numpy as np
# app = QtGui.QApplication([])

win = pg.GraphicsLayoutWidget(show=True, title="Basic plotting examples")
win.resize(1000,600)
win.setWindowTitle('pyqtgraph example: Plotting')

lbl = win.addLabel("<span style='font-size: 60pt'>label</span>")

plot = win.addPlot(title="courbe")

x = np.linspace(0, 50,1000)
y = np.cos(x)
curve = plot.plot(pen='y')
curve.setData(y)

proxy = QtGui.QGraphicsProxyWidget()
button = QtGui.QPushButton('Coucou')
proxy.setWidget(button)

def coucou_function():
    print("Hello World!")
    
win.addItem(proxy)#,row=1,col=1)
button.clicked.connect(coucou_function)