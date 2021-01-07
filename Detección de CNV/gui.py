import tkinter as tk

class Aplication(tk.Frame):

    def __init__(self):
        self.window = tk.Tk()
        super().__init__(self.window)

        self.window.title('Detector de CNV')
        self.window.configure(width=400, height=500, bg= '#E6FCE1')
        self.place(relwidth = 1, relheight = 1)
        self.window.resizable(False, False)
        self.configure(bg= '#E6FCE1')

        self.btnSel = tk.Button(self, text = 'Seleccionar archivo', font = ("", "14"))
        self.btnSel.place(x= 100, y= 10)

        self.textSel = tk.StringVar()
        self.textSel.set('')
        self.selName = tk.Label(self, textvariable = self.textSel, bg= '#E6FCE1', font = ("", "14"))
        self.selName.place(x= 100, y = 60)

        self.textGen = tk.StringVar()
        self.textGen.set('')
        self.selGen = tk.Label(self, textvariable=self.textGen, bg= '#E6FCE1', font = ("", "14"))
        self.selGen.place(x=100, y=430)