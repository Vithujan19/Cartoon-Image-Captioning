import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tqdm import tqdm
import yaml, csv, json


def load_config():
    with open('./config/config.yaml', 'r') as config_file:
        config = yaml.safe_load(config_file)
        config_file.close()
    return config


def get_dataset_split_as_df():
    '''
    load data split available as a csv file in directory
    '''
    config = load_config()
    data_split_dir = config['preprocessing']['train_test_val_split_csv']
    dataset_split = pd.read_csv(data_split_dir)
    return dataset_split


def get_image_to_caption_map(config, group):
    '''
    group = train / val / test
    obtain a dataframe containing image file names and their human annotated captions
    '''
    dataset_split_df = get_dataset_split_as_df()
    dataset_subset = [row for index, row in dataset_split_df.iterrows() if row['split'] == group]
    # print(len(dataset_subset))
    # print(dataset_subset[0]['image'], dataset_subset[0]['caption'])

    images_list = []
    captions_list = []
    images_dir = config['images_dir']

    for item in dataset_subset:
        # print("in loop", type(item['image']), type(item['caption']))
        # break
        filename = images_dir + item['image']
        captions = item['caption']
        images_list.append(filename)
        captions_list.append(captions)
    # print(len(images_list), len(captions_list))
    ret_df = pd.DataFrame({'filename': images_list, 'caption': captions_list})

    # add start token 'startseq' and end token 'endseq' to each caption
    ret_df['caption'] = ret_df['caption'].apply(lambda x: 'startseq ' + x + ' endseq')

    if config.get('experiment_size', None) is not None:
        ret_df = ret_df[:config['experiment_size']]

    return ret_df


def load_pretrained_embeddings(file_path):
    '''
    load Glove pretrained word vector embeddings from disk
    '''
    embeddings_index = {}
    print('\nembed file path = {}\n'.format(file_path))

    f = open(file_path, encoding="utf8")
    for line in tqdm(f):
        values = line.split(" ")
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    print("Found %s word vectors" % len(embeddings_index))
    return embeddings_index


def get_tokenizer_from_dir(config):
    ''''
    load the tokenizer file saved as json in disk. Used to convert words to tokens and vice-versa.
    '''
    # load tokenizer
    tokenizer_dir = config['preprocessing']['tokenizer_dir']
    with open(tokenizer_dir) as f:
        data = json.load(f)
        tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(data)
    return tokenizer


def plot_losses(filename):
    '''
    helper function to plot train and val losses per epochs
    '''
    loss_df = result = pd.read_csv(filename)

    plt.figure(figsize=(7, 4))
    plt.plot(range(1, len(loss_df)), loss_df['train_loss'], label='train_loss')
    plt.plot(range(1, len(loss_df)), loss_df['val_loss'], label='val_loss')
    plt.xticks(range(1, len(loss_df)))
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Per Epoch')
    plt.grid()
    plt.legend()
    plt.rcParams['figure.facecolor'] = 'white'
    plt.savefig('./data/loss_per_epoch_plot.png')
    plt.show()
    return
