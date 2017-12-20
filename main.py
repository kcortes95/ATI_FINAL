from tkinter import Tk, Entry, Scale, Label, LabelFrame, Button, Menu, filedialog, Canvas
from functions import *

subjects = ["", "Elvis Presley", "Kevin Cortes", "Lucas Casagrande"]


class ImageSelector:

    def __init__(self, master):
        self.master = master
        master.minsize(width=640, height=480)
        master.title("Adove Fotoyop")

        self.accept = Button(master, text="Predict Image", width=10, height=1, command=self.open)
        self.accept.pack()

    def open(self):
        filename = filedialog.askopenfilename(parent=root)
        test_img = cv2.imread(filename)
        predicted_img = predict(test_img, face_recognizer, subjects)
        print("Prediction complete")

        size_ratio = predicted_img.shape[0] / predicted_img.shape[1]
        cv2.imshow("Prediction", cv2.resize(predicted_img, (600, int(600 * size_ratio))))


root = Tk()
my_gui = ImageSelector(root)

root.iconbitmap('src/ati.ico')
my_gui.release = True
print("Preparing data...")
faces, labels = prepare_training_data("training-data")
print("Data prepared")

print("Total faces: ", len(faces))
print("Total labels: ", len(labels))

face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.train(faces, np.array(labels))

print("Ready to Predict")

root.mainloop()


