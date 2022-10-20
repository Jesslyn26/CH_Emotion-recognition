import numpy as np
import os as os
import keras
import cv2
from PIL import Image
import time
from multiprocessing import Pool

# filter all the warnings and error, also letting tensorflow use the CPU instead of GPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# creating a category for each emotion so it would display what type of emotion it is
emotion_names = ["Anger", "Disgust", "Fear", "Happiness", "Sadness", "Surprise"] 

# load the hierarchy data format file so it could predict the images
model_path = "emotion_model.h5"
loaded_model = keras.models.load_model(model_path)

# image path to process the image
img_path = r"C:\CH_Coursework2_Jesslyn Angela Chang\FER 2013\test\300 images/"
list_image = os.listdir(img_path)   


# predicting the emotion of an image
def emotion_recognition(images): 

    # reading each image and emotion in the file and resize it 
    image = cv2.imread(os.path.join(img_path, images))
    image_fromarray = Image.fromarray(image, 'RGB')
    resize_image = image_fromarray.resize((48,48))
    expand_input = np.expand_dims(resize_image, axis = 0)
    input_data = np.array(expand_input)
    input_data = input_data/255

    # predict and display the predicted image 
    pred = loaded_model.predict(input_data)
    result = pred.argmax()
    print(img_path + images)
    print(emotion_names[result])



if __name__ == '__main__':

    # counting the total number of processor in the device
    num_of_processors = os.cpu_count()

    # start timing the processed time until it successfully executed the whole program
    start_time = time.process_time()

    # mapping all the tasks to different processors
    with Pool(num_of_processors) as p: 
        p.map(emotion_recognition, list_image)

    # ending the process time
    end_time = time.process_time() - start_time

    # print the process time
    print(f"Time to process: {end_time} seconds")


    
    

    
    
    
        

    