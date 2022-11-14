import tkinter as tk
import time
import pandas as pd
from tkinter import ttk, filedialog
from tkinter import messagebox
from tkinter.ttk import Progressbar
from Code_Acquisition_Classes import acquisition_signal_capteur
from detection_default import detection_default
import matplotlib.pyplot as plt

class frame():

    def __init__(self):

        # Création fenetre principale

        self.app = tk.Tk()
        self.app.title("Maintenance prédictive")
        self.app.geometry("560x480")
        self.app.resizable(False, False)

        # Création de cadre

        self.OSframe = tk.LabelFrame(self.app, text="Système d'exploitation", )
        self.OSframe.pack(fill="x")

        self.time_frame = tk.LabelFrame(
            self.app, text="Temps d'acquisition (s)", )
        self.time_frame.pack(fill="x")


        self.acquis_frame = tk.LabelFrame(
            self.app, text="Nombre d'acquisitions")
        self.acquis_frame.pack(fill="x")
        
        self.method_frame = tk.LabelFrame(self.app, text="")
        self.method_frame.pack(fill="x")

        self.varOS = tk.IntVar()
        self.varMeth = tk.IntVar()
        self.varTime = tk.IntVar()
        self.varNbreAcquis = tk.IntVar()
        self.varF1 = tk.IntVar()
        self.varF2 = tk.IntVar()


        self.radio_widget_MacOS = tk.Radiobutton(
            self.OSframe, text="MacOS", value=0, variable=self.varOS)
        self.radio_widget_Windows = tk.Radiobutton(
            self.OSframe, text="Windows", value=1, variable=self.varOS)
        self.radio_widget_Linux = tk.Radiobutton(
            self.OSframe, text="Linux", value=2, variable=self.varOS)

        self.scale_w = tk.Scale(self.time_frame, from_=0, to=100, tickinterval=50,
                                orient="horizontal", length=200, variable=self.varTime)
        self.nbre_acquis = tk.Entry(
            self.acquis_frame, textvariable=self.varNbreAcquis)

        self.quit_button = tk.Button(self.app, text="Quit", command=self.fQuit)

        self.radio_widget_meth1 = tk.Button(
            self.method_frame, text="Analyse", command = self.show_next_window)
     

        self.radio_widget_MacOS.pack()
        self.radio_widget_Windows.pack()
        self.radio_widget_Linux.pack()

        self.nbre_acquis.pack()

        self.radio_widget_meth1.pack()
   

        self.scale_w.pack()

        self.quit_button.place(x=10, y=440)

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
        return self.methode1()
       
    def bar(self):
      
        labelAcq = tk.Label(self.next_window, text="Patientez")
        labelAcq.place(x = 360, y = 35)
        for acquisition in range(self.varNbreAcquis.get()):

            self.my_progress['value'] = (
                acquisition/self.varNbreAcquis.get())*100
            self.next_window .update_idletasks()
            time.sleep((self.varTime.get()))

    
        self.my_progress['value'] = 100
        labelTerm = tk.Label(self.next_window, text="Terminé !")
        labelTerm.place(x = 360, y = 35)
        #get_data = acquisition_signal_capteur(
        #T=self.varTime.get(), n=self.varNbreAcquis.get())
        self.analyse()
    def analyse(self):
        #analyse them#

        self.analyse_data = detection_default(
            duree=self.varTime.get(), nombre_signal=self.varNbreAcquis.get())

        self.analyse_data.pre_filtrage()
        plt.show()


    def suite_analyse(self):
        self.analyse_data.enveloppe(self.varF1.get(), self.varF2.get())
        plt.show() ##IMPORTANT !
    def methode1(self):
        self.next_window = tk.Toplevel(self.app)
        self.next_window.geometry("560x480")
        labelTime = tk.Label(
            self.next_window, text="Durée d'aquisition : " + str(self.varTime.get()) + " s")
        labelTime.pack()
        self.next_window.title("")
        self.my_progress = Progressbar(
            self.next_window, orient="horizontal", length=500, mode='determinate')

        tk.Button(self.next_window, text='Acquisition',
                  command=self.bar).pack(pady=10)

        # Faire un carré avec les frequence min et max à conserver.

        self.my_progress.pack()

        freq_frame = tk.LabelFrame(
            self.next_window, text="Sélectionner les fréquences f1 et f2 (Hz)")
        freq_frame.pack(fill="x")

        f1 = tk.Entry(freq_frame, textvariable=self.varF1)
        f2 = tk.Entry(freq_frame, textvariable=self.varF2)

        
        f1.pack()
        f2.pack()

        continue2_button = tk.Button(self.next_window, text="Détection enveloppe", command=self.suite_analyse)
        defaut_button = tk.Button(self.next_window, text="Detection défauts", command = self.tableau)
        continue2_button.pack()
        defaut_button.pack()

    def tableau(self):
        root = tk.Tk()
        root.title('Codemy.com - Excel To Treeview')
        root.iconbitmap('')
        root.geometry("700x500")
 
        my_frame = tk.Frame(root)
        my_frame.pack(pady=20)
        
        my_tree = ttk.Treeview(my_frame)
        
        def file_open():
            filename = filedialog.askopenfilename(
                initialdir="C://Users//Marien//Desktop//DPT",
                title = "Open A File",
                filetype=(("xlsx files", "*.xlsx"), ("All Files", "*.*"))
                )
        
            if filename:
                try:
                    filename = r"{}".format(filename)
                    df = pd.read_excel(filename)
                except ValueError:
                    my_label.config(text="File Couldn't Be Opened...try again!")
                except FileNotFoundError:
                    my_label.config(text="File Couldn't Be Found...try again!")
       
            clear_tree()
        
            # Set up new treeview
            my_tree["column"] = list(df.columns)
            my_tree["show"] = "headings"
            for column in my_tree["column"]:
                my_tree.heading(column, text=column)

            df_rows = df.to_numpy().tolist()
            for row in df_rows:
                my_tree.insert("", "end", values=row)
  
            my_tree.pack()
                
        def clear_tree():
            my_tree.delete(*my_tree.get_children())
    
        my_menu = tk.Menu(root)
        root.config(menu=my_menu)
        
        file_menu = tk.Menu(my_menu, tearoff=False)
        my_menu.add_cascade(label="Spreadsheets", menu=file_menu)
        file_menu.add_command(label="Open", command=file_open)
        
        my_label = tk.Label(root, text='')
        my_label.pack(pady=20)
        
        root.mainloop()
                
if __name__ == "__main__":
    f = frame()
