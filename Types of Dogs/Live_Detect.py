import tensorflow as tf
from keras.applications import imagenet_utils
from keras.models import load_model
from keras.preprocessing import image
import cv2, threading
import numpy as np

# cvtColor means coverting the selected frames color

model = load_model('Dog_n_cat_model_2.0.h5')
frame_to_predict = None
classification = True
graph = tf.get_default_graph()
label = ''
score = .0

class MyThread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        
    def run(self):
        global label
        global frame_to_predict
        global classification
        global model
        global graph
        
        with graph.as_default():

            while classification is True:
                if frame_to_predict is not None:
                    #frame_to_predict = cv2.cvtColor(frame_to_predict, cv2.COLOR_BGR2RGB).astype(np.float32)
                    #frame_to_predict = cv2.resize(frame_to_predict, (32, 32))
                    frame_to_predict = frame_to_predict.reshape((1, 32, 32, 3))
                    predictions = model.predict(frame_to_predict)

                    label=predictions[0]

keras_thread = MyThread()
keras_thread.start()

video_capture = cv2.VideoCapture(0) # Set to 1 for front camera
video_capture.set(4, 600) # Width
video_capture.set(5, 400) # Height


while (True):
    
    # Get the original frame from video capture
    ret, original_frame = video_capture.read()
    # Resize the frame to fit the imageNet default input size
    frame_to_predict = cv2.resize(original_frame, (32, 32))

    cv2.putText(original_frame, "Label: %s | Score: %.2f" % (label, score), 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the video
    cv2.imshow("Classification", original_frame)
    if (cv2.waitKey(1) & 0xFF == ord('q')):
        break