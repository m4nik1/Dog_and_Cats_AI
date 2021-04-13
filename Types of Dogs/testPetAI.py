import numpy as np 
from keras.models import load_model
from keras.preprocessing import image
from keras.utils import np_utils

classifier = load_model('Dog_n_cat_model_2.0.h5')


test_image = image.load_img('./Charlue.JPG', target_size=(64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
print(classifier.predict(test_image))


result = classifier.predict(test_image)
