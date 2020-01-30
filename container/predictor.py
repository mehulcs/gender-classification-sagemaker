import os
import json
import StringIO
import sys
import signal
import traceback
import flask
import pandas as pd
import cv2
import numpy as np
import cvlib
from flask import jsonify

from keras.preprocessing.image import img_to_array
from __future__ import print_function
from keras.models import load_model

WORKING_DIR = '/opt/ml/'
MODEL_FILE_PATH = os.path.join(WORKING_DIR, 'model', 'model.h5')
classes = ['man', 'woman']


class InferenceService(object):
    model = None                # Where we keep the model when it's loaded

    @classmethod
    def get_model(cls):
        if cls.model == None:
            cls.model = load_model('model.h5')
        return cls.model

    @classmethod
    def predict(cls, input):
        model = cls.get_model()
        return model.predict(input)


# The flask app for serving predictions
app = flask.Flask(__name__)


@app.route('/ping', methods=['GET'])
def ping():
    health = InferenceService.get_model() is not None

    status = 200 if health else 404
    return flask.Response(response='\n', status=status, mimetype='application/json')


@app.route('/invocations', methods=['POST'])
def transformation():
    data = None

    if flask.request.content_type != 'image/jpeg':
        return flask.Response(response='This predictor only supports JPEG Image', status=415, mimetype='text/plain')

    # Read image
    file = flask.request.files['image']
    fileData = file.read()
    image = cv2.imdecode(
        np.fromstring(fileData, np.uint8),
        cv2.IMREAD_COLOR
    )

    # Detect face in image
    faces, confidence = cvlib.detect_face(image)
    face = faces[0]
    (startX, startY) = face[0], face[1]
    (endX, endY) = face[2], face[3]

    # Crop face
    face_crop = np.copy(image[startY:endY, startX:endX])
    # preprocessing for gender detection model
    face_crop = cv2.resize(face_crop, (96, 96))
    face_crop = face_crop.astype("float") / 255.0
    face_crop = img_to_array(face_crop)
    face_crop = np.expand_dims(face_crop, axis=0)

    confidence = InferenceService.predict(faces)[0]
    index = np.argmax(confidence)

    return jsonify({'predictedClass': classes[index], 'confidence': confidence[index]})
