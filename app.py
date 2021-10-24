from flask import Flask, make_response, jsonify
from flask_restful import reqparse, abort
import pickle
# import numpy as np
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import re

app = Flask(__name__)

clf_path = './sentiment_analyser_keras.hdf5'
model = load_model(clf_path)


vec_path = 'tokenizer.pickle'
with open(vec_path, 'rb') as f:
    tokenizer = pickle.load(f)


# argument parsing
parser = reqparse.RequestParser()
parser.add_argument('query')


NON_ALPHANUM = re.compile(r'[\W]')
NON_ASCII = re.compile(r'[^a-z0-1\s]')


def normalize_texts(texts):
    normalized_texts = []
    for text in texts:
        lower = text.lower()
        no_punctuation = NON_ALPHANUM.sub(r' ', lower)
        no_non_ascii = NON_ASCII.sub(r'', no_punctuation)
        normalized_texts.append(no_non_ascii)
    return normalized_texts


@app.route('/', methods=['GET'])
def sa_trpp():
    # use parser and find the user's query
    args = parser.parse_args()
    user_query = args['query']

    MAX_LENGTH = 255

    user_query = normalize_texts([str(user_query)])
    uq_vectorized = tokenizer.texts_to_sequences(user_query)
    uq_vectorized = pad_sequences(uq_vectorized, maxlen=MAX_LENGTH)
    prediction = model.predict(uq_vectorized)
    pred_proba = prediction[0]

    # Output either 'Negative' or 'Positive' along with the score
    if pred_proba < 0.4:
        pred_text = 'Negative'
    elif pred_proba > 0.6:
        pred_text = 'Positive'
    else:
        pred_text = 'Neutral'

    # round the predict proba value and set to new variable
    confidence = round(pred_proba[0], 3)

    # create JSON object
    output = {'sentiment_prediction': pred_text,
              'sentiment_confidence': str(confidence)}
    response = make_response(jsonify(output))
    response.headers["Access-Control-Allow-Origin"] = "*"
    return response


if __name__ == '__main__':
    app.run(debug=False)
