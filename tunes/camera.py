
#importing req libs

import base64
import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense,Conv2D,MaxPooling2D,Flatten, BatchNormalization,Dropout
import matplotlib.pyplot as plt
from keras import callbacks
import numpy as np
from keras.applications.vgg16 import VGG16
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score,ConfusionMatrixDisplay
import seaborn as sns
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.callbacks import EarlyStopping
# from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import ImageDataGenerator





#data preprocessing 
batch_size=32
img_size=48
train_datagen = ImageDataGenerator(
                    rescale=1./255,
                    rotation_range=30,
                    shear_range=0.3,
                    #zoom_range=0.3,
                    width_shift_range = 0.1,
                    height_shift_range = 0.1,
                    horizontal_flip=True,
                    validation_split=0.3)

validation_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.3)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        'D:/kaggle/input/face-expression-recognition-dataset/images/train/',
        target_size=(img_size, img_size),
        color_mode =  'grayscale',
        batch_size=batch_size,
        class_mode='categorical',
        subset = 'training')

validation_generator = validation_datagen.flow_from_directory(
        'D:/kaggle/input/face-expression-recognition-dataset/images/train/',
        target_size=(img_size, img_size),
        color_mode =  'grayscale',
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation')

test_generator = test_datagen.flow_from_directory(
        'D:/kaggle/input/face-expression-recognition-dataset/images/validation/',
        target_size=(img_size, img_size),
        color_mode =  'grayscale',
        batch_size=batch_size,
        class_mode='categorical',
        shuffle = False)


train_generator.class_indices


#tuned model 
from keras.optimizers import Adam,SGD,RMSprop

initializer = tf.keras.initializers.HeUniform(seed=42)
reg = tf.keras.regularizers.L2(l2=0.01)
no_of_classes = 7

model_tuned = Sequential()

#1st CNN layer
model_tuned.add(Conv2D(64,(3,3),padding = 'same', activation='relu',input_shape = (48,48,1), kernel_initializer=initializer, kernel_regularizer=reg))
model_tuned.add(BatchNormalization())
model_tuned.add(MaxPooling2D(pool_size = (2,2)))
model_tuned.add(Dropout(0.25))

#2nd CNN layer
model_tuned.add(Conv2D(128,(5,5),padding = 'same', activation='relu', kernel_initializer=initializer, kernel_regularizer=reg))
model_tuned.add(BatchNormalization())
model_tuned.add(MaxPooling2D(pool_size = (2,2)))
model_tuned.add(Dropout (0.25))

#3rd CNN layer
model_tuned.add(Conv2D(512,(3,3),padding = 'same', activation='relu', kernel_initializer=initializer, kernel_regularizer=reg))
model_tuned.add(BatchNormalization())
model_tuned.add(MaxPooling2D(pool_size = (2,2)))
model_tuned.add(Dropout (0.25))

#4th CNN layer
model_tuned.add(Conv2D(512,(3,3), padding='same', activation='relu', kernel_initializer=initializer, kernel_regularizer=reg))
model_tuned.add(BatchNormalization())
model_tuned.add(MaxPooling2D(pool_size=(2, 2)))
model_tuned.add(Dropout(0.25))

model_tuned.add(Flatten())

#Fully connected 1st layer
model_tuned.add(Dense(256, activation='relu'))
model_tuned.add(BatchNormalization())
model_tuned.add(Dropout(0.25))


# Fully connected layer 2nd layer
model_tuned.add(Dense(512, activation='relu'))
model_tuned.add(BatchNormalization())
model_tuned.add(Dropout(0.25))

model_tuned.add(Dense(no_of_classes, activation='softmax'))

model_tuned.summary()



#applying call backs and compiling
 
from keras.optimizers import RMSprop,SGD,Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# checkpoint_tuned = ModelCheckpoint("./model.h5", monitor='val_acc', verbose=1, save_best_only=True, mode='max')
checkpoint_tuned = ModelCheckpoint("./model.keras", monitor='val_acc', verbose=1, save_best_only=True, mode='max')
early_stopping = EarlyStopping(monitor='val_loss',
                          min_delta=0,
                          patience=3,
                          verbose=1,
                          restore_best_weights=True
                          )

reduce_learningrate = ReduceLROnPlateau(monitor='val_loss',
                              factor=0.2,
                              patience=3,
                              verbose=1,
                              min_delta=0.0001)

callbacks_list_tuned = [checkpoint_tuned,reduce_learningrate]

epochs = 0

model_tuned.compile(loss='categorical_crossentropy',
              optimizer = Adam(learning_rate=0.0001),
              metrics=['accuracy'])


#fitting the model
history_tuned = model_tuned.fit(train_generator,
                                steps_per_epoch=train_generator.n//train_generator.batch_size,
                                epochs=epochs,
                                validation_data = validation_generator,
                                validation_steps = validation_generator.n//validation_generator.batch_size,
                                callbacks=callbacks_list_tuned
                                )






# def cam():
#     classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
#     face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    

#     

#         # Converting to grayscale image
#         gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#         # Apply face detection
#         cood = face_cascade.detectMultiScale(gray_frame)

#         # Loop through detected faces
#         for x, y, w, h in cood:
#             cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)

#             # Crop the detected face region
#             face_crop = np.copy(gray_frame[y:y+h, x:x+w])

#             if (face_crop.shape[0]) < 10 or (face_crop.shape[1]) < 10:
#                 continue

#             # Preprocessing for emotion detection model
#             face_crop = cv2.resize(face_crop, (48, 48))
#             face_crop = face_crop.astype("float") / 255.0
#             face_crop = img_to_array(face_crop)
#             face_crop = np.expand_dims(face_crop, axis=0)

#             # Apply emotion detection on face
#             conf = model_tuned.predict(face_crop)[0]  # model.predict returns a 2D matrix

#             # Get label with max accuracy
#             idx = np.argmax(conf)
#             label = classes[idx]

#             # Write label and confidence above face rectangle
#             cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX,
#                         0.7, (0, 255, 255), 2)

#             # Display output
#             cv2.imshow("Emotion Detection", frame)

#             # Return the detected emotion
#             webcam.release()
#             cv2.destroyAllWindows()
#             return label

#         # Display output
#         cv2.imshow("Emotion Detection", frame)

#         # Press "Q" to stop
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     # Release resources
#     webcam.release()
#     cv2.destroyAllWindows()
#     return None

def encode_image_to_base64(image_file):
    try:
        # Read the image file data
        image_data = image_file.read()
        
        # Decode the image data to a NumPy array
        np_arr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        # Check if the image is loaded properly
        if image is None:
            raise ValueError("Image decoding failed")

        # Encode the image to a base64 string
        success, buffer = cv2.imencode('.jpg', image)
        if not success:
            raise ValueError("Image encoding failed")
        
        image_bytes = buffer.tobytes()
        base64_string = base64.b64encode(image_bytes).decode('utf-8')
        return base64_string
    except Exception as e:
        print(f"Error encoding image to base64: {e}")
        return None

def cam(image_base64):
    try:
        classes = ['neutral', 'surprise', 'love', 'pop', 'fear', 'rock', 'sad']
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Decode the base64 image data
        image_data = base64.b64decode(image_base64)
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        
        # Apply face detection
        faces = face_cascade.detectMultiScale(image, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            face_crop = np.copy(image[y:y+h, x:x+w])

            if face_crop.shape[0] < 10 or face_crop.shape[1] < 10:
                continue

            # Preprocess the face crop for the model
            face_crop = cv2.resize(face_crop, (48, 48))
            face_crop = face_crop.astype("float") / 255.0
            face_crop = img_to_array(face_crop)
            face_crop = np.expand_dims(face_crop, axis=0)

            # Apply emotion detection on face
            conf = model_tuned.predict(face_crop)[0]

            # Get label with max confidence
            idx = np.argmax(conf)
            return idx  # Return the index of the predicted emotion

    except Exception as e:
        print(f"Error in cam function: {e}")
        return None