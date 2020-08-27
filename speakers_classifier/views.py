from django.shortcuts import render
from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response
import librosa
import numpy as np
import os

from .apps import PredictionConfig
from sklearn.preprocessing import StandardScaler
import urllib.request

ROOT_DIR = '/content'
number_of_mfcc = 40
frames = 87
genders = ['M','F']
duration = 2 #seconds
file_path = "wav1.wav"
# Process a sound file using MFCC
def get_mfcc_properties(signal_data,sr):
  signal = signal_data
  # pad signals of max seconds duration
  signal_length = duration * sr
  length = signal.shape[0]
  pad_length = signal_length - length
  if pad_length > 0:
    padding = np.zeros(pad_length)
    signal = np.hstack((signal, padding))
  mfcc_feats = librosa.feature.mfcc(signal, sr, n_mfcc=number_of_mfcc)

  return mfcc_feats


# Create your views here.
@api_view(['POST'])
def predict_gender(request):
    data = request.data
    file_url = data['filePath']
    response = urllib.request.urlopen(file_url)
    sound_file = response.read()
    with open(file_path, "wb") as file:
        file.write(sound_file)
    sound_data, sr = librosa.load(file_path, duration=duration)
    sc = StandardScaler()
    mfcc_data = sc.fit_transform(get_mfcc_properties(sound_data, sr))

    loaded_classifier = PredictionConfig.classifier
    pred = loaded_classifier.predict(np.reshape(mfcc_data,[1,number_of_mfcc,frames,1]))

    gender = genders[int(round(pred.reshape(-1)[0]))]

    response_dict = {"gender": gender}

    os.remove(file_path)

    return Response(response_dict, status=status.HTTP_201_CREATED)
