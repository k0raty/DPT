import tkinter as tk
from tkinter import messagebox
import time

### Création de fonctions

def fQuit():
    return app.destroy()

def show_error_window():
    messagebox.showerror("ERREUR", "Un problème est survenu !")

def show_next_window():
    if(varOS.get() == 0):
        return show_error_window()
    if(varOS.get() == 2):
        return show_error_window()

    next_window = tk.Toplevel(app)
    next_window.geometry("560x480")

    labelTime = tk.Label(next_window, text = "Durée d'aquisition : " + str(varTime.get()) + " s")
    labelTime.pack()

    if(varMeth.get() == 0):
        next_window.title("Méthode 1")
    if(varMeth.get() == 1):
        next_window.title("Méthode 2")
    if(varMeth.get() == 2):
        next_window.title("Méthode 3")





### Création fenetre principale

app = tk.Tk()
app.title("Maintenance prédictive")
app.geometry("560x480")
app.resizable(False, False)


### Création de cadre

OSframe = tk.LabelFrame(app, text="Système d'exploitation", )
OSframe.pack(fill = "x")

time_frame = tk.LabelFrame(app, text="Temps d'acquisition (s)", )
time_frame.pack(fill = "x")

method_frame = tk.LabelFrame(app, text="Méthode")
method_frame.pack(fill = "x")

### Création des variables

varOS = tk.IntVar()
varMeth = tk.IntVar()
varTime = tk.IntVar()

### Création widgets

radio_widget_MacOS = tk.Radiobutton(OSframe, text = "MacOS", value = 0, variable = varOS)
radio_widget_Windows = tk.Radiobutton(OSframe, text = "Windows", value = 1, variable = varOS)
radio_widget_Linux = tk.Radiobutton(OSframe, text = "Linux", value = 2, variable = varOS)

scale_w = tk.Scale(time_frame, from_ = 0, to = 100, tickinterval = 50, orient="horizontal", length = 200, variable = varTime)

continue_button = tk.Button(app, text = "Continue", command = show_next_window)

quit_button = tk.Button(app, text = "Quit", command = fQuit)

radio_widget_meth1 = tk.Radiobutton(method_frame, text = "Méthode 1", value = 0, variable = varMeth)
radio_widget_meth2 = tk.Radiobutton(method_frame, text = "Méthode 2", value = 1, variable = varMeth)
radio_widget_meth3 = tk.Radiobutton(method_frame, text = "Méthode 3", value = 2, variable = varMeth)


### Affichages

radio_widget_MacOS.pack()
radio_widget_Windows.pack()
radio_widget_Linux.pack()

radio_widget_meth1.pack()
radio_widget_meth2.pack()
radio_widget_meth3.pack()

scale_w.pack()

continue_button.place(x = 480, y = 440)
quit_button.place(x = 10, y = 440)


app.mainloop()