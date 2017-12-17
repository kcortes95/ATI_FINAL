from functions import *

subjects = ["", "Ramiz Raja", "Elvis Presley", "Kevin Cortes"]

print("Preparing data...")
faces, labels = prepare_training_data("training-data")
print("Data prepared")

print("Total faces: ", len(faces))
print("Total labels: ", len(labels))

face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.train(faces, np.array(labels))

print("Predicting images...")

test_img3 = cv2.imread("test-data/test3.jpg")

predicted_img3 = predict(test_img3, face_recognizer, subjects)
print("Prediction complete")

size_ratio = predicted_img3.shape[0] / predicted_img3.shape[1]
cv2.imshow(subjects[3], cv2.resize(predicted_img3, (600, int(600*size_ratio) )))
cv2.waitKey(0)

