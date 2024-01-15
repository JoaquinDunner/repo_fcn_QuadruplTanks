import os, sys, inspect
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == '__main__':
    meta = {}      

    #################################### Data de entrada ##############################################################
    ## n_inputs = canales de entrada
    n_inputs = 6
    ## hidden_size = número de dimensiones finales en el espacio latente h()
    hidden_size = 80
    ## batch_size = número de samples propagado por la red
    batch_size = 128
    ## seqlen = tamaño de ventana de tiempo tomada (en el paper = T-depth window)
    seqlen = 60 
    ## n_layer = cantidad de capas de encoder/decoder
    n_layers = 2
    ## epochs = ciclo de entrtenamiento en que pasa todo el dataset (un forward y un backward pass)
    epochs = 250
    ## beta = parámetro de trade-off de pérdidas -> 𝐿 = 𝐿𝐴𝐸 + 𝛽 𝐿𝑁𝐶𝐸 (22)
    beta = 1
    ## tau = parámetro de función random_seq_contrastive_loss
    tau = 1
    ## noise_power = varianza de ruido
    noise_powers = [1]
    ## noise_inputs = en función process_tanks() se entregan las inputs con ruido agragado (True) o no (False)
    noise_inputs = False
    ## bidirectional = propiedad de neuronas GRU: la secuencia se lee al derecho y al revés por dos capas distintas
    bidirectional = False
    ## missing_prob = DUDA! REVISAR, PARECE QUE NO SIRVE #####################################################################
    missing_prob = 0.2
    ## criterion = función de error, con posibilidades: DTW (Dynamic time warping), MSE o MAE
    criterion = 'MAE'
    ## data_type = qué datos queremos utilizar, con posibilidades: 'Tanks' o Else (Thickener)
    data_type = 'Tanks'
    ## missing = REVISAR, PARÁMETRO DE random_addlinear_noise
    missing = False
    ## mode = modos de la NCE Loss, con posibilidades:  'seq' -> Only Consecutive elements (Zc)
    ##                                                  'random' -> Only Random Selected elements (Zr)
    ##                                                  'combined' -> Combined elements (Zc + Zr)
    mode = 'combined'
    ## noise = tipo de ruido, con posibilidad: 'saltPepper' o else (no implementado)
    noise = 'saltPepper'
    ## plot = queremos plotear los resultados
    plot = True

    ## folder = carpeta donde se guardan todas las pruebas
    folder = 'fcn_AE_models/'

    ####################################################################################################################

    ## Testing = False -> entrenamiento
    ##         = True -> evaluación
    testing = False

    if not testing:
    	for noise_power in noise_powers:
            ## name = nombre de la subcarpeta donde se guarda la prueba
            name = f'uut_noise{noise_power}'

            meta = {'n_inputs': n_inputs, 'hidden_size': hidden_size, 'n_layers': n_layers, 'bidirectional': bidirectional, 
            'seqlen': seqlen, 'tau': tau, 'mode': mode, 'noise_power': noise_power, 'noise_inputs':noise_inputs,
            'batch_size': batch_size, 'name': name, 'missing': missing, 'data_type': data_type, 'criterion': criterion, 
            'missing_prob': missing_prob, 'beta': beta}

            print(f'Training {meta["name"]} with noise power = {noise_power}')
            meta['additional_info'] = 'En este caso no se hace la suma de para bypasear a la cnn' # ???
            ## Crear directorio (si es que no existe)
            current_directory = os.getcwd()
            folder_path = os.path.join(current_directory, folder)
            if not os.path.exists(folder_path):
                print(f'Directorio {folder} creado')
                os.makedirs(folder_path)
            ## Guardar metadata
            save_metadata(meta, '{}{}/'.format(folder, meta['name']))
            meta = load_metadata('{}{}/'.format(folder, meta['name']))
            
			## Carga y procesamiento de datos -> se llaman funciones dentro de TrainingAutoencoder()
            processor = DataProcessor(seqlen=meta['seqlen'])
            
            ## Crear modelo de CBDAE
            model = RNNAutoencoder_AddLayer(n_inputs=meta['n_inputs'], hidden_size=meta['hidden_size'], n_layers=meta['n_layers'],
											bidirectional=meta['bidirectional'], seqlen=meta['seqlen'],
											batch_size=meta['batch_size']).to(device)
            training_autoencoder = TrainingAutoencoder(processor, model, data_type=meta['data_type'], shuffle_train=True, noise_inputs=meta['noise_inputs'],
													folder='{}{}/'.format(folder, meta['name']), noise=noise, noise_power=meta['noise_power'],
													path='model', batch_size=meta['batch_size'], missing_prob=meta['missing_prob'])
            training_autoencoder.train(epochs=epochs, checkpoints=[], beta=meta['beta'], 
									missing=meta['missing'], scheduler=False, n_inputs=meta['n_inputs'],
									criterion=meta['criterion'], tau=meta['tau'], seqlen=meta['seqlen'], mode=meta['mode'])
            # break
    else:
        ## Ojo, cambié el número de componentes PCA a 10 (eran 20 originalmente, línea 471)
        plot = False
        folders = os.listdir(folder)
        folders = [elem for elem in folders if 'fig' not in elem and 'baseline' not in elem and 'svg' not in elem]
        for name in folders:
            meta = load_metadata('{}{}/'.format(folder, name))
            print(f'Testing {meta["name"]} with beta = {meta["beta"]}')
            
            model = RNNAutoencoder_AddLayer(n_inputs=meta['n_inputs'], hidden_size=meta['hidden_size'],
					n_layers=meta['n_layers'],
					bidirectional=meta['bidirectional'], seqlen=meta['seqlen'],
					batch_size=meta['batch_size']).to(device)
            
            model.load_state_dict(torch.load('{}{}/model'.format(folder, meta['name']), map_location=torch.device('cpu')))
            
            processor = DataProcessor(seqlen=meta['seqlen'])
            
            training_autoencoder = TrainingAutoencoder(processor, model, data_type=meta['data_type'], shuffle_train=True, noise_inputs=meta['noise_inputs'],
						folder='{}{}/'.format(folder, meta['name']), noise=noise, noise_power=meta['noise_power'],
						path='model', batch_size=meta['batch_size'], missing_prob=meta['missing_prob'])
            
            training_autoencoder.eval(missing=meta['missing'], n_inputs=meta['n_inputs'], plot=plot, name=meta['name'])