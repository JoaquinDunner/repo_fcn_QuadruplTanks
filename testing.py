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
from utils.data_processor import generate_noise
import pprint

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == '__main__':
    meta = {}      

    #################################### Data de entrada ##############################################################
    ## n_inputs = canales de entrada
    n_inputs = 6
    ## hidden_size = número de dimensiones finales en el espacio latente h()
    hidden_size = 80
    ## batch_size = número de samples propagado por la red
    batch_size = 64
    ## seqlen = tamaño de ventana de tiempo tomada (en el paper = T-depth window)
    seqlen = 60 
    ## n_layer = cantidad de capas de encoder/decoder
    n_layers = 2
    ## epochs = ciclo de entrtenamiento en que pasa todo el dataset (un forward y un backward pass)
    epochs = 200
    ## beta = parámetro de trade-off de pérdidas -> 𝐿 = 𝐿𝐴𝐸 + 𝛽 𝐿𝑁𝐶𝐸 (22)
    beta = 1
    ## tau = parámetro de función random_seq_contrastive_loss
    tau = 1
    ## noise_power = varianza de ruido
    noise_powers = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4]
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
    missing = True
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
    	
    ## Ojo, cambié el número de componentes PCA a 10 (eran 20 originalmente, línea 471)
    plot = False
    folders = os.listdir(folder)
    folders = [elem for elem in folders if 'fig' not in elem and 'baseline' not in elem and 'svg' not in elem]
    for name in folders:
        meta = load_metadata('{}{}/'.format(folder, name))
        print(f'Testing {meta["name"]} with noise power = {meta["noise_power"]}')
          
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
        
        ## Importar resultados
        print(f'Load pickle from {folder}{name}/reconstruction_results.pkl')
        reconstruct_result = torch.load(f'{folder}{name}/reconstruction_results.pkl')
        
        ## Noisy signal
        for i in range(n_inputs):
            if i == 0:
                X_clean_full = reconstruct_result[f'{i+1}']['X_clean']
            else:
                X_clean_full = np.vstack((X_clean_full,reconstruct_result[f'{i+1}']['X_clean']))

        Y_noisy = generate_noise(X_clean_full, multiplier=1, noise_inputs=False, plot=False)
        print(Y_noisy.shape)

        for i in range(1, meta['n_inputs'] + 1):
            ## Skip inputs (index 1 and 2)
            if i <= 2: 
                pass
            
            else:
                ## Numpy arrays
                X_clean = reconstruct_result[f'{i}']['X_clean']
                Y_pred = reconstruct_result[f'{i}']['Y_pred']
                total_rmse = reconstruct_result[f'{i}']['total_rmse']
                lost_index_rmse = reconstruct_result[f'{i}']['lost_index_rmse']
                lost_index_rmse_no_missing = reconstruct_result[f'{i}']['lost_index_rmse_no_missing']
                cos_similarity = reconstruct_result[f'{i}']['cosine_similarity']
                X_in_no_missing = reconstruct_result[f'{i}']['X_in_no_missing']
                Y_pred_no_missing = reconstruct_result[f'{i}']['Y_pred_no_missing']

                upto = 3000
                plt.figure(1)
                plt.subplot(int(f'32{i-2}'))
                # plt.plot(Y_pred_no_missing[:upto], label='Y_pred_no_missing')
                plt.plot(Y_noisy[i-1, :upto], label='Y_noisy')
                plt.plot(Y_pred[:upto], label='Y_pred')
                plt.plot(X_clean[:upto], label='Y_clean')
                # plt.plot(X_in_no_missing[:upto], label='X_in_no_missing')
                plt.legend(loc='upper right')
                plt.title(f'Output {i}')

                # plt.figure(2)
                # plt.subplot(int(f'32{i}'))
                # plt.plot(X_clean[:upto], label='X_clean')
                # plt.plot(X_in_no_missing[:upto], label='X_in_no_missing')
                # plt.legend()
                # plt.title(f'Indice {i}')

        # plt.savefig(f"{folder}{name}/{meta['noise_power']}.png")
        plt.show()
        