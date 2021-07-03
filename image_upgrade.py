import cv2
from cv2 import dnn_superres
import tkinter
from tkinter import BooleanVar, messagebox

class MainWindow(tkinter.Frame):
    def __init__(self, parent) -> None:
        super(MainWindow, self).__init__(parent)
        self.parent = parent
        self.grid(row=0, column=0, sticky=tkinter.NSEW)

        # Read the desired model
        self.model_map = {
            2: (BooleanVar(), 'EDSR_x2.pb'),
            3: (BooleanVar(), 'EDSR_x3.pb'),
            4: (BooleanVar(), 'EDSR_x4.pb'),
        }

        self.init_ui()

    def init_ui(self) -> None:
        from tkinter import Button, Label, Checkbutton

        button_input = Button(self, text='Open image', command=self.set_input)
        button_input.focus_set()
        self.label_input = Label(self, relief=tkinter.SUNKEN)
        label_scale = Label(self, text='Scales:', relief=tkinter.GROOVE)
        checkb2 = Checkbutton(self, text='x2', variable=self.model_map[2][0])
        checkb3 = Checkbutton(self, text='x3', variable=self.model_map[3][0])
        checkb4 = Checkbutton(self, text='x4', variable=self.model_map[4][0])
        button_do = Button(self, text='Upgrade', command=self.do)
        self.label_do = Label(self, relief=tkinter.GROOVE)
        button_quit = Button(self, text='Quit', command=self.quit)

        button_input.grid(row=0, column=0, padx=3, pady=3, sticky=tkinter.NSEW)
        self.label_input.grid(row=0, column=1, columnspan=3,
                              padx=3, pady=3, sticky=tkinter.NSEW)
        label_scale.grid(row=1, column=0, padx=3, pady=3, sticky=tkinter.NSEW)
        checkb2.grid(row=1, column=1, padx=3, pady=3, sticky=tkinter.NSEW)
        checkb3.grid(row=1, column=2, padx=3, pady=3, sticky=tkinter.NSEW)
        checkb4.grid(row=1, column=3, padx=3, pady=3, sticky=tkinter.NSEW)
        button_do.grid(row=2, column=0, padx=3, pady=3, sticky=tkinter.NSEW)
        self.label_do.grid(row=2, column=1, columnspan=3,
                              padx=3, pady=3, sticky=tkinter.NSEW)
        button_quit.grid(row=3, column=3, padx=3, pady=3, sticky=tkinter.NSEW)

        self.columnconfigure(0, weight=0)
        self.columnconfigure(1, weight=1)
        self.columnconfigure(2, weight=1)
        self.columnconfigure(3, weight=1)

    def quit(self, event=None) -> None:
        self.parent.destroy()

    def set_input(self) -> None:
        from tkinter import filedialog as fd
        self.label_input['text'] = fd.askopenfilename()

    def do(self) -> None:
        from tkinter.messagebox import showinfo
        from os.path import splitext

        # Read image
        finput = self.label_input['text']
        if not finput:
            showinfo(title='Image upgrade', message='No input image')

        fname, fext = splitext(finput)[0], splitext(finput)[1]
        image = cv2.imread(finput)

        for scale, item in self.model_map.items():
            if not item[0].get():
                continue

            fmodel = item[1]
            self.label_do['text'] = f'scale {scale}'
            self.update()

            # Create an SR object
            sr = dnn_superres.DnnSuperResImpl_create()
            sr.readModel(fmodel)

            # Set the desired model and scale to get correct pre- and post-processing
            sr.setModel("edsr", scale)

            # Upscale the image
            result = sr.upsample(image)

            # Save the image
            cv2.imwrite(f"{fname}_x{scale}{fext}", result)

        cv2.destroyAllWindows()
        showinfo(title='Upgrade image', message='Done')


if __name__ == '__main__':
    app = tkinter.Tk()
    app.title('Image upgrade')
    app.minsize(width=300, height=150)
    app.resizable(True, False)
    app.eval('tk::PlaceWindow . center')
    app.rowconfigure(0, weight=1)
    app.columnconfigure(0, weight=1)
    mw = MainWindow(app)
    app.protocol('WM_DELETE_WINDOW', mw.quit)
    app.mainloop()
