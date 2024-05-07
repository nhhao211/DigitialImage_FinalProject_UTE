import cv2
import numpy as np
import os

# Load user IDs from labels.txt
with open('labels.txt', 'r') as f:
    lines = f.readlines()
    num_users = len(lines)

data = [] 
label = []

# Loop through each user's data directory
for user_id in range(1, num_users + 1):
    user_data_dir = f'./data/{user_id}/'
    if not os.path.exists(user_data_dir):
        continue
    
    # Loop through each image
    for i in range(1, 201):
        filename = f'{user_data_dir}anh{i}.jpg'
        try:
            Img = cv2.imread(filename)
            if Img is None:
                print(f"Warning: Unable to read image at {filename}")
                break  

            Img = cv2.cvtColor(Img, cv2.COLOR_BGR2GRAY)
            Img = cv2.resize(src=Img, dsize=(100, 100))
            Img = np.array(Img)
            data.append(Img)
            label.append(user_id - 1)
        except Exception as e:
            print("Error processing image:", filename)
            print(e)
data = np.array(data)
label = np.array(label)
data = data.reshape(-1, 100, 100, 1)
X_train = data / 255

from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical

le = LabelEncoder()
trainY = le.fit_transform(label)
num_classes = len(le.classes_)
trainY = to_categorical(trainY, num_classes=num_classes)

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation

Model = Sequential()
Model.add(Conv2D(32, (3, 3), padding="same", input_shape=(100, 100, 1)))
Model.add(Activation("relu"))
Model.add(Conv2D(32, (3, 3), padding="same"))
Model.add(Activation("relu"))
Model.add(MaxPooling2D(pool_size=(2, 2)))
Model.add(Conv2D(64, (3, 3), padding="same"))
Model.add(Activation("relu"))
Model.add(MaxPooling2D(pool_size=(2, 2)))
Model.add(Flatten())
Model.add(Dense(512))
Model.add(Activation("relu"))
Model.add(Dense(num_classes))
Model.add(Activation("softmax"))
Model.summary()

Model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print("Start training")
Model.fit(X_train, trainY, batch_size=128, epochs=20)
Model.save("khuonmat.h5")
