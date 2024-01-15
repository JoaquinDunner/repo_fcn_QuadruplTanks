import os, sys, inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, parent_dir)
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from sklearn.decomposition import PCA
from nets import RNNAutoencoder, RNNAutoencoder_AddLayer
from utils.nce_loss import random_seq_contrastive_loss
from utils.data_processor import DataProcessor
import os
import copy
import matplotlib.pyplot as plt
import time
from torch.optim.lr_scheduler import LambdaLR
import json
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from random_addlinear_noise import *
from testing_AE import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == '__main__':
    ## folder = carpeta donde se guardan todas las pruebas
    folder = 'fcn_AE_models/'
    ## noise = tipo de ruido, con posibilidad: 'saltPepper' o else (no implementado)
    noise = 'saltPepper'
    	
    ## Ojo, cambié el número de componentes PCA a 10 (eran 20 originalmente, línea 471)
    plot = False
    #folders = os.listdir(folder)
    #folders = [elem for elem in folders if 'fig' not in elem and 'baseline' not in elem and 'svg' not in elem]
    #for name in folders:
    ## Folder name del modelo
    name = 'uut_noise1'

    meta = load_metadata('{}{}/'.format(folder, name))
    print(f'Testing {meta["name"]} with noise power = {meta["noise_power"]}')
        
    model = RNNAutoencoder_AddLayer(n_inputs=meta['n_inputs'], hidden_size=meta['hidden_size'],
            n_layers=meta['n_layers'],
            bidirectional=meta['bidirectional'], seqlen=meta['seqlen'],
            batch_size=meta['batch_size']).to(device)
    
    model.load_state_dict(torch.load('{}{}/model'.format(folder, meta['name']), map_location=torch.device('cpu')))
    
    processor = DataProcessor(seqlen=meta['seqlen'])
    
    testing_autoencoder = TestingAutoencoder(processor, model, data_type=meta['data_type'], shuffle_train=True, noise_inputs=meta['noise_inputs'],
                folder='{}{}/'.format(folder, meta['name']), noise=noise, noise_power=meta['noise_power'],
                path='model', batch_size=meta['batch_size'], missing_prob=meta['missing_prob'])
    

    testing_autoencoder.eval(missing=meta['missing'], n_inputs=meta['n_inputs'], plot=plot, name=meta['name'])
        