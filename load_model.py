# Importing libraries
from tensorflow import keras

def load_model(path:str):
    '''
    Function: Loads the keras model which is the best amongst the 10 variants given by faculty

    Input:-
    path [str]: Path of the model

    Output:-
    model [Keras Sequential]: Sequential Model (Simple Neural Network)
    '''
    model = keras.models.load_model(path) # Model is loaded
    return model # Model is returned