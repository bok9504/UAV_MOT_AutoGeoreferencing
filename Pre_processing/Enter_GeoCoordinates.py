import tkinter.font
import tkinter


class Tk_GeoPoint(tkinter.Tk):
    def __init__(self, parent):
        tkinter.Tk.__init__(self,parent)
        self.parent = parent
        self.pointList = None
        self.initialize()

    def initialize(self):
        font=tkinter.font.Font(size=12)

        self.label1 = tkinter.Label(text="longitude : ")
        self.label1.grid(column=0, row=0)

        self.label2 = tkinter.Label(text="latitude  : ")
        self.label2.grid(column=0, row=1)

        self.longitude = tkinter.Text(width = 25, height=1, font=font)
        self.longitude.grid(column=1, row=0)

        self.latitude = tkinter.Text(width = 25, height=1, font=font)
        self.latitude.grid(column=1, row=1)

        self.btn = tkinter.Button(text='Enter', command=self.btncmd)
        self.btn.grid(column=0, row=2, columnspan=2)

    def btncmd(self):
        x = self.longitude.get("1.0",'end-1c')
        y = self.latitude.get("1.0",'end-1c')
        self.pointList = (float(x),float(y))

        if self.pointList=="":
            Win2=tkinter.Tk()
            Win2.withdraw()

        self.destroy()


if __name__ == '__main__':
    app = Tk_GeoPoint(None)
    app.title('Enter GeoCoordinates')
    app.geometry('310x80')
    app.mainloop()
    print(app.pointList)