import tensorflow
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPool2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.preprocessing.image import ImageDataGenerator 


# Assinging the path into the objects
ds_train = r"C:\CH_Coursework2_Jesslyn Angela Chang\FER 2013\train"
ds_test = r"C:\CH_Coursework2_Jesslyn Angela Chang\FER 2013\test"
ds_val = r"C:\CH_Coursework2_Jesslyn Angela Chang\FER 2013\val"


# Preparing image to be process
train_data = ImageDataGenerator(rescale = 1./225, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)
val_data = ImageDataGenerator(rescale = 1./255)
train_set = val_data.flow_from_directory(ds_train, target_size = (48,48), batch_size = 64, class_mode = "categorical")
val_set = val_data.flow_from_directory(ds_val, target_size = (48,48), batch_size = 64, class_mode = "categorical")


# Creating the CNN architecture
emotions_model = Sequential()

emotions_model.add(Conv2D(16, (3, 3), input_shape = (48, 48, 3), activation = 'relu'))
emotions_model.add(MaxPool2D(pool_size = (2,2)))
emotions_model.add(Conv2D(64, (3,3), activation = 'relu'))
emotions_model.add(Dropout(0.25))

emotions_model.add(Conv2D(128, (3, 3), activation = 'relu'))
emotions_model.add(MaxPool2D(pool_size = (2,2)))
emotions_model.add(Conv2D(128, (3,3), activation = 'relu'))
emotions_model.add(Dropout(0.25))


emotions_model.add(Flatten())
emotions_model.add(Dense(units = 1024, activation = 'relu'))
emotions_model.add(Dropout(0.50))
emotions_model.add(Dense(units = 6, activation = 'softmax'))

emotions_model.compile(optimizer = "adam", loss = 'categorical_crossentropy', metrics = ['accuracy'])

# to see the summary of the model -> emotions_model.summary()

#Training the data
training = emotions_model.fit(train_set, epochs = 50, validation_data = val_set)
emotions_model.save('emotion_model.h5')
