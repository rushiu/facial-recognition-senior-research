'''
2168/2168 [==============================] - 236s 109ms/step - loss: 0.0548 - acc: 0.9843
 - val_loss: 1.4564 - val_acc: 0.7596

'''
from keras.layers import Conv2D, MaxPooling2D, Activation, Dense, Flatten
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils

import numpy as np
import cv2, glob, os  # only used for loading the image, you can use anything that returns the image as a np.ndarray
from PIL import Image
#from skutils import shuffle --need to download


#######Load in DATA
'''
What to do:
1. load in all images, seperate into training (280) vs. testing data (120)
2. reshape it
3. make the validation data 
'''
X_train = np.array([cv2.imread("./patchesF/s1/1p1.jpg").flatten()])
X_test = np.array([cv2.imread("./patchesF/s1/1p1.jpg").flatten()])
X_demo = np.array([cv2.imread("./patchesF/s41/1p1.jpg").flatten()])


tes = np.array(cv2.imread("./patchesF/s1/1p1.jpg"))
print(tes.shape)
y_train = np.ones((2224,), dtype = int) #used to be 2168 (5-21)
y_test = np.ones((960,), dtype = int) #used to be 936 (5-21)

trainCount = 0;
testCount = 0
for image_path in glob.glob("./patchesF/*/*.jpg"):
    imName = image_path.split("/")
    person = int(imName[2][1:])
    testORtrain = int(imName[3][:-6])

    if testORtrain < 8:
    	n = np.array(cv2.imread(image_path)).flatten()
    	X_train = np.append(X_train, [n], axis=0)
    	y_train[trainCount] = person
    	trainCount += 1

    else:
    	n = np.array(cv2.imread(image_path)).flatten()
    	X_test = np.append(X_test, [n], axis=0)
    	y_test[testCount] = person
    	testCount += 1
print(trainCount, testCount)


X_train = np.delete(X_train, (0), axis=0)
X_test = np.delete(X_test, (0), axis=0)

###   Shuffle data -- download library and implement later   ###
#X_train, y_train = shuffle(X_train, y_train, random_state = 2)
#X_test, y_test = shuffle(X_train, y_train, random_state = 2)

X_train = X_train.reshape(2224, 3, 228, 188) #used to be 2168 (5-21)
X_test = X_test.reshape(960  , 3, 228, 188) #used to be 2168 (5-21)

X_demo = X_demo.reshape(1, 3, 228, 188)


X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train /= 255
X_test /= 255

y_train = np_utils.to_categorical(y_train, 42) #used to be 41 (5-21)
y_test = np_utils.to_categorical(y_test, 42) #used to be 41 (5-21)

print(X_train[0].shape)

# 3 filters in both conv layers
model = Sequential()
model.add(Conv2D(64, kernel_size=3, input_shape= (3, 228, 188), activation='relu')) 

model.add(Conv2D(32, (3, 3), activation='relu', data_format='channels_first'))
model.add(Conv2D(16, (3, 3), activation='relu', data_format='channels_first'))
model.add(Conv2D(8, (3, 3), activation='relu', data_format='channels_first'))
model.add(Conv2D(4, (3, 3), activation='relu', data_format='channels_first'))


# Lets activate then pool!

model.add(Flatten())

#Full Connection
model.add(Dense(units=128, activation='relu'))
model.add(Dense(units=42, activation='softmax')) #used to be 41 (5-21)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10) #finish this & run


print("Testing-------->")
print("Inputting a picture of Mom (Folder 41)")

result = model.predict(X_demo)
print(result)
print("This is a picture of:")
if result[0,41] == 1:
    print("||\\  //||    ---   ||\\  //||")
    print("|| \\// ||  ||   || || \\// ||")
    print("||      ||    ---   ||      ||")

#remember-->> face one is at index 1 & so on for the results, NOT index 0!!


#https://www.youtube.com/watch?v=2pQOXjpO_u0



