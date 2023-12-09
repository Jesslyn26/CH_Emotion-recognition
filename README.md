# Emotion Recognition
An emotion recognition program that can predict emotions (happy, sad, angry, disgust, and fear) based on the inputted dataset. The program is implementing deep learning (convolutional neural network/ CNN) by utilizing TensorFlow, Keras, Numpy, and Pandas. The purpose of this project is to see the impact of computer hardware (CPU and GPU) towards the computer's performance
  
## File description
- FER 2010 = The dataset that is used to train, validate, and test the model
- Training_Emotion-recognition = Pre-process and compile the machine learning model
- emotion_model.h5 = the file to store CNN model or machine learning model in Tensorflow
- multiprocessEmotionRecognition = multiprocess the prediction
- sequentialEmotionRecognition = sequentially predict images 

## Layers that was used to build the CNN model
- convolutional array (2D) = filter and multiplying the neighboring values or elements into a 2-d inputs (3x3)
- Max pooling = selecting the maximum elements for each region 
- Dropout = deactivate a portion of inputs or nodes during training
- Flatten = converting data to 1 dimenton in order to feed it to the next layer
- Dense = connecting all of the node or neurons that recieve input as one output (happy, sad, angry, disgust, and fear)
