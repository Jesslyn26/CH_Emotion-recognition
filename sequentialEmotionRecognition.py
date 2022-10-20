import os
import keras
import numpy as np
import cv2
from PIL import Image
import time

# Skipping all the CPU and warnings and errors
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


if __name__ == '__main__':
    # loading model and the emotional category to be displayed
    emotion_names = ["Anger", "Disgust", "Fear", "Happiness", "Sadness", "Surprise"]
    model_path = "emotion_model.h5"
    loaded_model = keras.models.load_model(model_path)


    # setting the image path so image can be loaded
    set_of_images = r"C:\CH_Coursework2_Jesslyn Angela Chang\FER 2013\test\300 images"
    i = 0

    # starting the 
    start_time = time.process_time()

    while i < len(os.listdir(set_of_images)):
         
        img_path = set_of_images +"/"+ os.listdir(set_of_images)[i]
        
        image = cv2.imread(img_path)

        image_fromarray = Image.fromarray(image, 'RGB')
        resize_image = image_fromarray.resize((48,48))
        expand_input = np.expand_dims(resize_image, axis = 0)
        input_data = np.array(expand_input)
        input_data = input_data/255

        pred = loaded_model.predict(input_data)
        result = pred.argmax()
        print(img_path)
        print(emotion_names[result])
        i += 1

    end_process_time = time.process_time() - start_time

    print(f"Time to process: {end_process_time} seconds")