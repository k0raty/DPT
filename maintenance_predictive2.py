import tkinter as tk
from tkinter import messagebox
from tkinter.ttk import Progressbar

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


    if(varMeth.get() == 0):
        return methode1()
    if(varMeth.get() == 1):
        next_window.title("Méthode 2")
    if(varMeth.get() == 2):
        next_window.title("Méthode 3")


def methode1():
    next_window = tk.Toplevel(app)
    next_window.geometry("560x480")
    labelTime = tk.Label(next_window, text = "Durée d'aquisition : " + str(varTime.get()) + " s")
    labelTime.pack()
    next_window.title("Méthode 1")
    my_progress = Progressbar(next_window, orient="horizontal",length=500,mode='determinate')

    def bar():
        import time
        for acquisition in range(1, varNbreAcquis.get()):

            my_progress['value'] = (acquisition/varNbreAcquis.get())*100
            next_window .update_idletasks()
            time.sleep((varTime.get()))
            labelAcq = tk.Label(next_window, text = "Acquisition " + str(acquisition))
            labelAcq.pack()

        my_progress['value'] = 100
        labelAcq = tk.Label(next_window, text = "Acquisition " + str(acquisition))
        labelAcq.pack()
        label = tk.Label(next_window, text = "Terminé")
        label.pack()

    tk.Button(next_window, text = 'Start', command = bar).pack(pady = 10)
    my_progress.pack()


def methode2():
    next_window = tk.Toplevel(app)
    next_window.geometry("560x480")
    labelTime = tk.Label(next_window, text = "Durée d'aquisition : " + str(varTime.get()) + " s")
    labelTime.pack()
    next_window.title("Méthode 2")

def methode3():
    next_window = tk.Toplevel(app)
    next_window.geometry("560x480")
    labelTime = tk.Label(next_window, text = "Durée d'aquisition : " + str(varTime.get()) + " s")
    labelTime.pack()
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
varNbreAcquis = tk.IntVar()

### Création widgets

radio_widget_MacOS = tk.Radiobutton(OSframe, text = "MacOS", value = 0, variable = varOS)
radio_widget_Windows = tk.Radiobutton(OSframe, text = "Windows", value = 1, variable = varOS)
radio_widget_Linux = tk.Radiobutton(OSframe, text = "Linux", value = 2, variable = varOS)

scale_w = tk.Scale(time_frame, from_ = 0, to = 100, tickinterval = 50, orient="horizontal", length = 200, variable = varTime)
nbre_acquis = entry = tk.Entry(time_frame, textvariable = varNbreAcquis)

continue_button = tk.Button(app, text = "Continue", command = show_next_window)

quit_button = tk.Button(app, text = "Quit", command = fQuit)

radio_widget_meth1 = tk.Radiobutton(method_frame, text = "Méthode 1", value = 0, variable = varMeth)
radio_widget_meth2 = tk.Radiobutton(method_frame, text = "Méthode 2", value = 1, variable = varMeth)
radio_widget_meth3 = tk.Radiobutton(method_frame, text = "Méthode 3", value = 2, variable = varMeth)


### Affichages

radio_widget_MacOS.pack()
radio_widget_Windows.pack()
radio_widget_Linux.pack()

nbre_acquis.pack()

radio_widget_meth1.pack()
radio_widget_meth2.pack()
radio_widget_meth3.pack()

scale_w.pack()

continue_button.place(x = 480, y = 440)
quit_button.place(x = 10, y = 440)


app.mainloop()