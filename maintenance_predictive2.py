import tkinter as tk
from tkinter import messagebox
from tkinter.ttk import Progressbar
from Code_Acquisition_Classes import acquisition_signal_capteur
from detection_default import detection_default
### Création de fonctions

class frame():
    
    def __init__(self):

       ### Création fenetre principale

       self.app = tk.Tk()
       self.app.title("Maintenance prédictive")
       self.app.geometry("560x480")
       self.app.resizable(False, False)


       ### Création de cadre

       self.OSframe = tk.LabelFrame(self.app, text="Système d'exploitation", )
       self.OSframe.pack(fill = "x")

       self.time_frame = tk.LabelFrame(self.app, text="Temps d'acquisition (s)", )
       self.time_frame.pack(fill = "x")

       self.method_frame = tk.LabelFrame(self.app, text="Méthode")
       self.method_frame.pack(fill = "x")

       ### Création des variables

       self.varOS = tk.IntVar()
       self.varMeth = tk.IntVar()
       self.varTime = tk.IntVar()
       self.varNbreAcquis = tk.IntVar()
       self.varF1 = tk.IntVar()
       self.varF2 = tk.IntVar()

       ### Création widgets

       self.radio_widget_MacOS = tk.Radiobutton(self.OSframe, text = "MacOS", value = 0, variable = self.varOS)
       self.radio_widget_Windows = tk.Radiobutton(self.OSframe, text = "Windows", value = 1, variable = self.varOS)
       self.radio_widget_Linux = tk.Radiobutton(self.OSframe, text = "Linux", value = 2, variable = self.varOS)

       self.scale_w = tk.Scale(self.time_frame, from_ = 0, to = 100, tickinterval = 50, orient="horizontal", length = 200, variable = self.varTime)
       self.nbre_acquis = entry = tk.Entry(self.time_frame, textvariable = self.varNbreAcquis)

       self.continue_button = tk.Button(self.app, text = "Continue", command = self.show_next_window)

       self.quit_button = tk.Button(self.app, text = "Quit", command = self.fQuit)

       self.radio_widget_meth1 = tk.Radiobutton(self.method_frame, text = "Méthode 1", value = 0, variable = self.varMeth)
       self.radio_widget_meth2 = tk.Radiobutton(self.method_frame, text = "Méthode 2", value = 1, variable = self.varMeth)
       self.radio_widget_meth3 = tk.Radiobutton(self.method_frame, text = "Méthode 3", value = 2, variable = self.varMeth)


       ### Affichages

       self.radio_widget_MacOS.pack()
       self.radio_widget_Windows.pack()
       self.radio_widget_Linux.pack()

       self.nbre_acquis.pack()

       self.radio_widget_meth1.pack()
       self.radio_widget_meth2.pack()
       self.radio_widget_meth3.pack()

       self.scale_w.pack()

       self.continue_button.place(x = 480, y = 440)
       self.quit_button.place(x = 10, y = 440)


       self.app.mainloop()
    
    def fQuit(self):
        return self.app.destroy()
    
    def show_error_window(self):
        messagebox.showerror("ERREUR", "Un problème est survenu !")
    
    def show_next_window(self):
        if(self.varOS.get() == 0):
            return self.show_error_window()
        if(self.varOS.get() == 2):
            return self.show_error_window()
    
    
        if(self.varMeth.get() == 0):
            return self.methode1()
        if(self.varMeth.get() == 1):
            self.next_window.title("Méthode 2")
        if(self.varMeth.get() == 2):
            self.next_window.title("Méthode 3")
    
    def bar(self):
        """
        progress bar

        Returns
        -------
        None.

        """
        import time
        for acquisition in range(1, self.varNbreAcquis.get()):

            self.my_progress['value'] = (acquisition/self.varNbreAcquis.get())*100
            self.next_window .update_idletasks()
            time.sleep((self.varTime.get()))
            labelAcq = tk.Label(self.next_window, text = "Acquisition " + str(acquisition))
            labelAcq.pack()

        self.my_progress['value'] = 100
        labelAcq = tk.Label(self.next_window, text = "Acquisition " + str(acquisition))
        labelAcq.pack()
        label = tk.Label(self.next_window, text = "Terminé")
        label.pack()
        
    def methode1(self):
        self.next_window = tk.Toplevel(self.app)
        self.next_window.geometry("560x480")
        labelTime = tk.Label(self.next_window, text = "Durée d'aquisition : " + str(self.varTime.get()) + " s")
        labelTime.pack()
        self.next_window.title("Méthode 1")
        self.my_progress = Progressbar(self.next_window, orient="horizontal",length=500,mode='determinate')
        
        
    
        tk.Button(self.next_window, text = 'Start', command = self.bar).pack(pady = 10)
        
        #get the data#
        get_data =  acquisition_signal_capteur(self.varTime.get(),self.varNbreAcquis.get())
        
        #analyse them#
        analyse_data = detection_default(duree = self.varTime.get(),nombre_signal=self.varNbreAcquis.get())
        analyse_data.method_envelop()
        F1  = tk.Entry(self.next_window, textvariable = self.varF1)
        F1.pack()
        F2  = tk.Entry(self.next_window, textvariable = self.varF2)
        F2.pack()
        analyse_data.method_envelop_2(self.varF1.get(),self.varF2.get())
        #Faire un carré avec les frequence min et max à conserver.
        self.fQuit()
        
       # my_progress.pack()
    
    
    def methode2(self):
        self.next_window = tk.Toplevel(self.app)
        self.next_window.geometry("560x480")
        labelTime = tk.Label(self.next_window, text = "Durée d'aquisition : " + str(self.varTime.get()) + " s")
        labelTime.pack()
        self.next_window.title("Méthode 2")
    
    def methode3(self):
        self.next_window = tk.Toplevel(self.app)
        self.next_window.geometry("560x480")
        labelTime = tk.Label(self.next_window, text = "Durée d'aquisition : " + str(self.varTime.get()) + " s")
        labelTime.pack()
        self.next_window.title("Méthode 3")


