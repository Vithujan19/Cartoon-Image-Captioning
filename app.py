from flask import Flask, request, jsonify, render_template
import flask
app = Flask(__name__)
from PIL import Image
from module.Trainer import Trainer
from utils.load_data_utils import load_config
import numpy as np
import os


@app.route('/')
def index():
    return flask.render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():

    config = load_config()
    max_len = 21
    vocab_size = 8619
    filtered_embedding_dir = './data/filtered_embed_vocabsize{}.csv'.format(vocab_size)

    embedding_matrix = np.genfromtxt(filtered_embedding_dir, delimiter=',')

    trainer = Trainer(config, vocab_size, max_len, embedding_matrix)

    images = request.files.getlist('photo')
    if not images:
        print('No files uploaded')
        return 'No files uploaded', 400

    predictions = []

    for image in images:
        file_name = image.filename
        img_path = 'sample_images/' + file_name
        prediction = trainer.predict_caption('static/' + img_path, beam_search_pred=True)
        predictions.append({
            'image_path': img_path,
            'caption': prediction
        })
    return render_template('caption.html', predictions=predictions)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8081)
