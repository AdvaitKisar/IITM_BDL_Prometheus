# Importing libraries
from tensorflow.keras import Sequential
import numpy as np

def predict_digit(model:Sequential, data_point:list):
    '''
    Function: Predicts the digit using model and image

    Input:-
    model [Keras Sequential]: Sequential Model (Simple Neural Network)
    data_point [list]: Flattened image

    Output:-
    digit [str]: Predicted digit
    '''
    output = model.predict(data_point) # Output is generated which is 10 dimensional
    digit = str(np.argmax(output, axis=1)[0]) # Index with highest probability is the prediction
    return digit # Digit is returned