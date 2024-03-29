import os
from numpy.core.numeric import True_
from module.Preprocessor import Preprocessor
from module.Trainer import Trainer
from utils.load_data_utils import load_config
from utils.img_features_utils import load_image_features_on_the_fly


if __name__ == '__main__':
    config = load_config()

    '''
    #1. create a directory to store model checkpoints and results
    '''
    if not os.path.exists(config['store']):
        os.mkdir(config['store'])
        os.mkdir(config['store'] + config['nn_params']['pred_df_dir'])

    '''
    #2. preprocessing: obtain raw images and captions and convert them into tensorflow format
    '''
    pp = Preprocessor(config)
    train_dataset, val_dataset = pp.prep_data()

    print('shape of embedding_matrix =', pp.embedding_matrix.shape)
    print('vocab_size = ', pp.vocab_size) 
    print('cardinality of train_dataset = ', train_dataset.cardinality().numpy())
    print('cardinality of val_dataset = ', val_dataset.cardinality().numpy())

    for element in train_dataset:
        print('shape of elements = {}, {}'.format(element[0].shape, element[1].shape))
        break
    print('-'*100)
    
    '''
    #3. training
    '''
    trainer = Trainer(config, pp.vocab_size, pp.max_len, pp.embedding_matrix)
    trainer.initiate_training(
        train_dataset, 
        val_dataset, 
        load_from_checkpoint=True, 
        load_loss_file=True, 
        save_loss_to_dir=True,
    )

    '''
    #4. metrics evaluation
    '''
    # bleu_df = trainer.compute_bleu_scores(config, group='val', search_method='greedy', save_to_dir=True)
    # bleu_df = trainer.compute_bleu_scores(config, group='train', search_method='beam')
    bleu_df = trainer.compute_bleu_scores(config, group='val', search_method='beam')

    # one-off predictions
    filename = 'data/sample_images/110595925_f3395c8bd6.jpg'
    img_tensor_val = load_image_features_on_the_fly(filename)
    pred_cap, _ = trainer.greedy_search_pred(img_tensor_val)
    print(f'\npredicted caption for single image = {pred_cap}\n')
