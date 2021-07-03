import cv2
from cv2 import dnn_superres
import tkinter
from tkinter import BooleanVar, Variable, messagebox


class MainWindow(tkinter.Frame):
    def __init__(self, parent) -> None:
        super(MainWindow, self).__init__(parent)
        self.parent = parent
        self.grid(row=0, column=0, sticky=tkinter.NSEW)

        self.model_name = 'edsr'
        # Read the desired model
        self.model_map = {
            2: BooleanVar(),
            3: BooleanVar(),
            4: BooleanVar(),
        }

        self.init_ui()

    def init_ui(self) -> None:
        from tkinter import Button, Label, Checkbutton, Spinbox, Entry

        self.do_scale = BooleanVar()
        self.do_enhance = BooleanVar()

        button_input = Button(self, text='Open image', command=self.set_input)
        button_input.focus_set()
        self.label_input = Label(self, relief=tkinter.SUNKEN)
        checkb_scale = Checkbutton(
            self, text='Scales:', variable=self.do_scale)
        checkb2 = Checkbutton(self, text='x2', variable=self.model_map[2])
        checkb3 = Checkbutton(self, text='x3', variable=self.model_map[3])
        checkb4 = Checkbutton(self, text='x4', variable=self.model_map[4])
        checkb_enh = Checkbutton(self, text='Enhance: try sigma_s from 0 to 200 and sigma_r from 0.0 to 1.0',
                                 variable=self.do_enhance)
        label_sigmas = Label(self, text='sigma_s:',
                             relief=tkinter.GROOVE, anchor=tkinter.E)
        self.spinbox_ss = Spinbox(self, from_=0, to=200)
        label_sigmar = Label(self, text='sigma_r:',
                             relief=tkinter.GROOVE, anchor=tkinter.E)
        self.entry_sr = Entry(self, text='0.0')
        button_do = Button(self, text='Upgrade', command=self.do)
        self.label_do = Label(self, relief=tkinter.GROOVE)
        button_quit = Button(self, text='Quit', command=self.quit)

        button_input.grid(row=0, column=0, padx=3, pady=3, sticky=tkinter.NSEW)
        self.label_input.grid(row=0, column=1, columnspan=3,
                              padx=3, pady=3, sticky=tkinter.NSEW)
        checkb_scale.grid(row=1, column=0, padx=3, pady=3, sticky=tkinter.NSEW)
        checkb2.grid(row=1, column=1, padx=3, pady=3, sticky=tkinter.NSEW)
        checkb3.grid(row=1, column=2, padx=3, pady=3, sticky=tkinter.NSEW)
        checkb4.grid(row=1, column=3, padx=3, pady=3, sticky=tkinter.NSEW)
        checkb_enh.grid(row=2, column=0, columnspan=4,
                        padx=3, pady=3, sticky=tkinter.W)
        label_sigmas.grid(row=3, column=0, padx=3, pady=3, sticky=tkinter.NSEW)
        self.spinbox_ss.grid(row=3, column=1, padx=3,
                             pady=3, sticky=tkinter.NSEW)
        label_sigmar.grid(row=3, column=2, padx=3, pady=3, sticky=tkinter.NSEW)
        self.entry_sr.grid(row=3, column=3, padx=3,
                           pady=3, sticky=tkinter.NSEW)
        button_do.grid(row=4, column=0, padx=3, pady=3, sticky=tkinter.NSEW)
        self.label_do.grid(row=4, column=1, columnspan=3,
                           padx=3, pady=3, sticky=tkinter.NSEW)
        button_quit.grid(row=5, column=3, padx=3, pady=3, sticky=tkinter.NSEW)

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

        if self.do_scale.get():
            for scale, checked in self.model_map.items():
                if not checked.get():
                    continue

                fmodel = f'{self.model_name}_x{scale}.pb'
                self.label_do['text'] = f'scale {scale}'
                self.update()

                # Create an SR object
                sr = dnn_superres.DnnSuperResImpl_create()
                sr.readModel(fmodel)

                # Set the desired model and scale to get correct pre- and post-processing
                sr.setModel(self.model_name, scale)

                # Upscale the image
                up_image = sr.upsample(image)
                # Save the image
                cv2.imwrite(f"{fname}_x{scale}{fext}", up_image)

                # Enhancing
                if self.do_enhance.get():
                    self.label_do['text'] = 'enhancing'
                    self.update()
                    enh_image = cv2.detailEnhance(
                        up_image, sigma_s=float(self.spinbox_ss.get()), sigma_r=float(self.entry_sr['text']))
                    cv2.imwrite(f"{fname}_x{scale}_enh{fext}", enh_image)
        elif self.do_enhance.get():
            # Only enhancing
            self.label_do['text'] = 'enhancing'
            self.update()
            enh_image = cv2.detailEnhance(
                image, sigma_s=float(self.spinbox_ss.get()), sigma_r=float(self.entry_sr.get()))
            cv2.imwrite(f"{fname}_enh{fext}", enh_image)

        cv2.destroyAllWindows()
        self.label_do['text'] = 'done'
        showinfo(title='Image upgrade', message='Done')


if __name__ == '__main__':
    app = tkinter.Tk()
    app.title('Image upgrade')
    app.minsize(width=400, height=150)
    app.resizable(True, False)
    app.eval('tk::PlaceWindow . center')
    app.rowconfigure(0, weight=1)
    app.columnconfigure(0, weight=1)
    mw = MainWindow(app)
    app.protocol('WM_DELETE_WINDOW', mw.quit)
    app.mainloop()
