from django.apps import AppConfig
import os
import librosa
import numpy as np
from keras.models import model_from_json
from sklearn.preprocessing import StandardScaler


ROOT_DIR = '/content'
number_of_mfcc = 40
frames = 87
genders = ['M','F']
duration = 2 #seconds
def load_model_json(dir_path,model_name):
  json_file = open(os.path.join(dir_path,(model_name + '.json')), 'r')
  loaded_model_json = json_file.read()
  json_file.close()
  loaded_model = model_from_json(loaded_model_json)
  loaded_model.load_weights(os.path.join(dir_path, (model_name + '.h5')))

  return loaded_model

class PredictionConfig(AppConfig):
    name = 'Prediction'
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    CLASSIFIER_FOLDER = os.path.join(BASE_DIR, os.path.join('speakers_classifier','classifiers'))
    #CLASSIFIER_FILE = CLASSIFIER_FOLDER / "IRISRandomForestClassifier.joblib"
    CLASSIFIER_FILE = "mfcc_model_2"
    classifier = load_model_json(CLASSIFIER_FOLDER,CLASSIFIER_FILE)


class SpeakersClassifierConfig(AppConfig):
    name = 'speakers_classifier'
