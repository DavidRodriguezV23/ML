from tkinter import filedialog, Label
from PIL import ImageTk
from PIL import Image
from utils import load_image
import numpy as np

class Controller():

    def __init__(self, view, model):
        self.model = model
        self.view = view
        self.view.btnSel['command'] = self.select
        self.view.mainloop()

    def select(self):
        self.view.window.filename = filedialog.askopenfilename(initialdir='C:\ ',
                                                               title='Seleccione la imagen a ser analizada',
                                                               filetypes=(
                                                                   ('Todos los archivos', '*.*'), ('jpg', '*.jpg'),
                                                                   ('png', '*.png'),
                                                                   ('jpeg', '*.jpeg')))

        self.view.textSel.set(self.view.window.filename.split('/')[-1].split('.')[0])
        image = Image.open(self.view.window.filename)
        w, h  = image.size
        image = image.resize((300, 300))
        self.view.image = ImageTk.PhotoImage(image)
        self.view.image_label = Label(image = self.view.image)
        self.view.image_label.place(x= 50, y= 100)

        X = load_image(self.view.window.filename)
        y = self.predict(X)
        if np.argmax(y, axis= 1)[0] == 0:
            self.view.selGen.place(x=100, y=430)
            self.view.textGen.set('Clasificación: Normal')
        else:
            self.view.selGen.place(x=12, y=430)
            self.view.textGen.set('Clasificación: Neovascularización coroidea')

    def predict(self, X):
        return self.model.predict(X)