from keras.models import load_model
import numpy as np
from data_helpers import load_data, clean_str

def pad_sentences(sentence):
    sentence = clean_str(sentence)
    sentence = sentence.split(" ")
    padding_word = "<PAD/>"

    num_padding = 56 - len(sentence)
    new_sentence = sentence + [padding_word] * num_padding

    return new_sentence

# load vocal
x, y, vocal, vocal_inv = load_data()

# load model
model = load_model('weights.051-0.7638.hdf5')


from flask import Flask
from flask import request
from flask import jsonify

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def api_call():
    data = request.get_json()
    sentences = data["sentences"]

    pred_arr = []
    for s in sentences:
        padded_sentence = pad_sentences(s)
        x = np.array([[vocal.get(word, "0") for word in padded_sentence]])
        result = model.predict(x, batch_size=1)
        if (result[0, 0] > result[0, 1]):
            pred_arr.insert(len(pred_arr), 'NEGATIVE')
        else:
            pred_arr.insert(len(pred_arr), 'POSITIVE')

    return jsonify(results=pred_arr)


if __name__ == "__main__":
    app.run()
