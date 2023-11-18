from module.Preprocessor import Preprocessor
from module.Trainer import Trainer
from utils.load_data_utils import load_config
import numpy as np
import os

config = load_config()
pp = Preprocessor(config)
max_len = 21
vocab_size = 8619
filtered_embedding_dir = './data/filtered_embed_vocabsize{}.csv'.format(vocab_size)

embedding_matrix = np.genfromtxt(filtered_embedding_dir, delimiter=',')

trainer = Trainer(config, vocab_size, max_len, embedding_matrix)
img_path = 'data/sample_images/110595925_f3395c8bd7.jpg'
prediction = trainer.predict_caption(img_path, beam_search_pred=True)
print(prediction)


