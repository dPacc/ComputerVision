import cv2
import numpy as np
import pickle


face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
# The Face Recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")

labels = {}
with open("labels.pickle", 'rb') as f:
    og_labels = pickle.load(f)
    labels = {v:k for k,v in og_labels.items()}


cap = cv2.VideoCapture(0)

def main():
    while(True):
        # Capture video frame by frame
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
        for (x, y, w, h) in faces:
            # Locating the face on the video capture
            #print(x, y, w, h)
            # Region of Interest and saving only the location where the face is
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]

            # How to recognize? You could use a Deep Learning Model
            # Head over to the faces-train.py to train a Model
            id_, conf = recognizer.predict(roi_gray)
            if conf >= 45: #and conf <= 85:
                # print(id_)
                # print(labels[id_])
                font = cv2.FONT_HERSHEY_SIMPLEX
                name = labels[id_]
                color = (255, 255, 255)
                stroke = 2
                cv2.putText(frame, name, (x, y), font, 1, color, stroke, cv2.LINE_AA)


            image_item = "my_image.png"
            cv2.imwrite(image_item, roi_gray)

            # Draw a rectangle
            color = (255, 0, 0)
            stroke = 4
            endcord_x = x + w
            endcord_y = y + h
            cv2.rectangle(frame, (x, y), (endcord_x, endcord_y), color, stroke)

        # Display the resulting frame
        cv2.imshow('Frame', frame)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    # When everything is done, release the Capture
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
