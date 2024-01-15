import pandas as pd
import numpy as np
import torch
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error
import random
import colorednoise as cn
import copy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def openFile(name):
    with open(name, 'rb') as file:
        return pickle.load(file)

def saveFile(obj, name):
    with open(name, 'wb') as file:
        pickle.dump(obj, file)


def generate_noise(data_clean, multiplier=1, noise_inputs=False, plot=False):
    """
    Generates Salt and Pepper Noise
    """
    if noise_inputs:
        ############################ Por qué *30 ?
        data_clean[:, :2] = data_clean[:, :2]*30 
    ## noise_inputs = False -> inputs (columnas 0 y 1) se separan de mediciones7data_clean (columnas 2, 3, 4 y 5)    
    else:
        inputs = data_clean[:, :2]
        #print(f'inputs = {inputs}')
        data_clean = data_clean[:, 2:]
        #print(f'data_clean = {data_clean}')

    
    scaler = StandardScaler() ## StandardScaler() = Standardize features by removing the mean and scaling to unit variance
    scaler.fit(data_clean)
    scales = scaler.scale_ ## Per feature relative scaling of the data to achieve zero mean and unit variance
    #print(f'scales = {scales}')

    ## beta = dependiendo de él, es borwn noise (beta=2), pink noise (beta=1), etc. -> https://en.wikipedia.org/wiki/Colors_of_noise
    beta = 0  # the exponent

    #### White Noises

    samples = data_clean.shape[0] ## samples = filas o muestras del dataset
    ## cn.powerlaw_psd_gaussian = Generate Gaussian distributed noise with a power law spectrum with arbitrary exponents (https://github.com/felixpatzelt/colorednoise)
    ## -> devuelve una lista con el número de columnas de data_clean np.arrays (matriz de data_clean.shape[1] x data_clean.shape[0])
    y_white = [cn.powerlaw_psd_gaussian(beta, samples) for i in range(data_clean.shape[1])]
    #print(f'y_white.shape = {y_white}')
    ## Pasar de nuevo a matriz de 4 columnas y multiplicar por el noise_power (=multiplier)
    y_white = np.array(y_white).transpose() * multiplier

    #### Salt and Pepper noise
    mult = [0.5, 0.7, 1, 1.2, 1.5] ## Multiplicadores escogidos para contaminar sample
    p = 0.05 ## Probabilidad de seleccionar una sample para contaminar
    data_salt_and_pepper = copy.deepcopy(data_clean + y_white)
    signs = [-1, 1] ## Signo de la contaminación

    for j in range(data_clean.shape[0]):
        ## Se escogen aleatoriamente multiplicadores, signos y samples (95% no será contaminado)
        selected_mults = np.array([random.choice(mult) for j in range(data_clean.shape[1])])
        selected_probs = np.array([int(random.random() < p) for j in range(data_clean.shape[1])])
        selected_signs = np.array([random.choice(signs) for j in range(data_clean.shape[1])])
        ## El producto de scales * [resto] entrega el ruido para cada fila y luego se suma a la copia ruidosa de [data_clean + y_white]
        addition = scales * selected_mults * selected_probs * selected_signs
        data_salt_and_pepper[j, :] += addition

    ## Eliminamos casos negativos, asignando 0 a la medición
    data_salt_and_pepper[np.where(data_salt_and_pepper < 0)] = 0

    if not noise_inputs:
        ## Si no hay noise_inputs, concatenamos nuevamente los inputs a las mediciones contaminadas con ruido (blanco + saltPepper)
        data_salt_and_pepper = np.concatenate([inputs, data_salt_and_pepper], axis=1)
    else:
        data_salt_and_pepper[:, :2] = data_salt_and_pepper[:, :2]/30

    if plot:
        plt.figure()
        plt.title(multiplier)
        plt.plot(data_salt_and_pepper[1000:1300, 2])
        plt.grid()
        plt.show()
    return data_salt_and_pepper



class DataProcessor():
    """
    Process data from the thickener and from the tanks
    """
    def __init__(self, seqlen=100):
        self.seqlen = seqlen
        self.tickener_signals = ['br_7120_ft_1002', 'bj_7110_ft_1012' , 'bg_7110_dt_1011_solido', 'bk_7110_ft_1030',
            'bp_7110_ot_1003', 'bo_7110_lt_1009_s4', 'bq_7110_pt_1010', 'bi_7110_dt_1030_solido']


    def make_batch(self, data, seqlen=60, cpu=False, torch_type=True, shuffle=True):
        data = np.array(data)

        scaler = RobustScaler(quantile_range=(10, 90))

        data = scaler.fit_transform(data)
        data_out = []
        for i in range(len(data) - seqlen):
            data_out.append(data[i: i + seqlen, :])

        data_out = np.array(data_out)

        def shuffle_in_unison_scary(a, b):
            rng_state = np.random.get_state()
            np.random.shuffle(a)
            np.random.set_state(rng_state)
            np.random.shuffle(b)

        if shuffle:
            np.random.shuffle(data_out)

        if torch_type:
            data_out = torch.from_numpy(data_out)
            if not cpu:
                data_out = data_out.to(device)


        return data_out.float(), scaler



    def make_batch_tanks(self, data, data_preproc, seqlen=60, cpu=False, torch_type=True, shuffle=True):
        """
                       Window creation
                       In this case each window overlaps in T-1 points with consecutive windows
        """
        scaler = StandardScaler()
        scaler_preproc = StandardScaler()

        data = scaler.fit_transform(data)
        data_preproc = scaler_preproc.fit_transform(data_preproc)

        data_out = []
        data_out_preproc = []
        for i in range(len(data) - seqlen):
            data_out.append(data[i: i + seqlen, :])
            data_out_preproc.append(data_preproc[i: i + seqlen, :])

        data_out = np.array(data_out)
        data_out_preproc = np.array(data_out_preproc)

        def shuffle_in_unison_scary(a, b):
            rng_state = np.random.get_state()
            np.random.shuffle(a)
            np.random.set_state(rng_state)
            np.random.shuffle(b)

        if shuffle:
            np.random.shuffle(data_out)
            np.random.shuffle(data_out_preproc)

        if torch_type:
            data_out = torch.from_numpy(data_out)
            data_out_preproc = torch.from_numpy(data_out_preproc)

            if not cpu:
                data_out = data_out.to(device)
                data_out_preproc = data_out_preproc.to(device)

        return data_out.float(), scaler, data_out_preproc.float(), scaler_preproc


    ## data = datos contaminados con ruido blanco+saltPepper ; data_preproc = datos limpios originales
    def make_batch_new_data_tanks(self, data, data_preproc, seqlen=60, cpu=False, torch_type=True, shuffle=True):
        """
               Window creation
               In this case each window doesn't overlap with other ones
        """
        data = np.array(data)

        scaler = StandardScaler()
        scaler_preproc = StandardScaler()

        ## Devuelve los datos transformados: una lista de data.shape[0] np.arrays, cada uno con data.shape[0] elementos (cada fila pasa a ser un elemento de la lista)
        data = scaler.fit_transform(data)
        #print(f'make_batch_new_data_tanks: len(data) = {data}')
        data_preproc = scaler_preproc.fit_transform(data_preproc)

        ## Creamos una nueva lista, con data.shape[0]/seqlen np.arrays, cada uno de con seqlen secuencias de largo data.shape[1]
        data_out = []
        data_out_preproc = []
        i = 0
        while i < len(data) - seqlen:
            data_out.append(data[i: i + seqlen, :])
            data_out_preproc.append(data_preproc[i: i + seqlen, :])
            i += seqlen
        #print(f'data_out.len = {len(data_out)}')
        #print(f'data_out_preproc.len = {len(data_out_preproc)}')

        ## Recupero el arreglo en forma de tensor: data.shape[0]/seqlen x seqlen x n_inputs
        data_out = np.array(data_out)
        data_out_preproc = np.array(data_out_preproc)
        #print(f'data_out.shape = {(data_out.shape)}')
        #print(f'data_out_preproc.shape = {(data_out_preproc.shape)}')

        ## Esta función NO se usa
        def shuffle_in_unison_scary(a, b):
            rng_state = np.random.get_state()
            np.random.shuffle(a)
            np.random.set_state(rng_state)
            np.random.shuffle(b)

        ## Mezclar las columas (mezcla sobre data.shape[0]/seqlen)
        if shuffle:
            np.random.shuffle(data_out)
            np.random.shuffle(data_out_preproc)

        ## Convertir de numpy a torch los datos
        if torch_type:
            data_out = torch.from_numpy(data_out)
            data_out_preproc = torch.from_numpy(data_out_preproc)
            if not cpu:
                data_out = data_out.to(device)
                data_out_preproc = data_out_preproc.to(device)

        return data_out.float(), scaler, data_out_preproc.float(), scaler_preproc


    def make_batch_new_data(self, data, seqlen=60, cpu=False, torch_type=True, shuffle=True):
        """
                     Window creation
                     In this case each window doesn't overlap with other ones
        """
        data = np.array(data)

        scaler = RobustScaler()

        ## Normalización de datos según RobustScaler(), más detalles en función anterior
        data = scaler.fit_transform(data)

        ## Creamos una nueva lista, con data.shape[0]/seqlen np.arrays, cada uno de con seqlen secuencias de largo data.shape[1]        
        data_out = []
        i = 0
        while i < len(data) - seqlen:
            data_out.append(data[i: i + seqlen, :])
            i += seqlen

        ## Recuperar forma del tensor: data.shape[0]/seqlen x seqlen x n_inputs
        data_out = np.array(data_out)

        ## Esta función NO se utiliza
        def shuffle_in_unison_scary(a, b):
            rng_state = np.random.get_state()
            np.random.shuffle(a)
            np.random.set_state(rng_state)
            np.random.shuffle(b)

        ## Mezclar las columas (mezcla sobre data.shape[0]/seqlen)
        if shuffle:
            np.random.shuffle(data_out)

        ## Convertir de numpy a torch los datos
        if torch_type:
            data_out = torch.from_numpy(data_out)
            if not cpu:
                data_out = data_out.to(device)

        return data_out.float(), scaler


    def process_tickener(self, folder='../Thickener_Data/Thickener/', signals=[], ratio=0.8, shuffle=True, new_data=False):
        parameters = openFile(folder + 'parametrosNew.pkl')

        if signals == []:
            signals = self.tickener_signals

        n_inputs = len(signals)

        # Loading Data
        data = pd.read_pickle(folder + 'dcs_data_04_18_2018_TO_26_02_2019_5min.pkl')[signals]


        # Parameters
        centers = np.array([parameters[tag]['center'] for tag in data.columns]).reshape(1, -1)
        scales = np.array([parameters[tag]['scale'] for tag in data.columns]).reshape(1, -1)

        #Limits
        inf_limit = centers - 1*scales
        inf_limit[np.where(inf_limit < 0)] = 0
        sup_limit = centers + 1*scales

        # Physical Restrictions
        def limits_func(variables):
            variables[np.where(variables > sup_limit)] = np.nan
            variables[np.where(variables < inf_limit)] = np.nan
            return variables

        # Apply Physiscal Restictions
        columns = data.columns
        data = pd.DataFrame(limits_func(np.array(data)), columns=columns)

        # NA values filling
        data = data.fillna(method='bfill').fillna(method='ffill')

        if new_data:
            data, scaler = self.make_batch_new_data(data=data, seqlen=self.seqlen, cpu=True,
                                shuffle=shuffle)
        else:
            data, scaler = self.make_batch(data=data, seqlen=self.seqlen, cpu=True, shuffle=shuffle)

        length = data.shape[0]
        data_train = data[:int(length*ratio), :, :]
        data_test = data[int(length*ratio):, :, :]

        data_dict = {'train_data': data_train, 'test_data': data_test, 'scaler': scaler, 'signals': signals}

        return data_dict


    def process_tanks(self, folder='../Tanks_Data/No_Noised_Inputs/', signals=[], ratio=[0.65, 0.2, 0.15], shuffle=True, new_data=True,
                      type_of_noise='white', noise_power=1, noise_inputs=False):

        ratio_train = ratio[0]
        ratio_val = ratio[1]
        ratio_test = ratio[2]

        #### Loading Data
        data_preproc = torch.load(folder + 'data_clean.pkl') # Original data to compare with output data
        #print(f'data_preproc = {data_preproc}')
        print(f'data_preproc.shape = {data_preproc.shape}')
        if type_of_noise == 'saltPepper':
            print('{} - {}'.format(type_of_noise, noise_power))
            ## Data tiene los datos, con ruido blanco+saltPepper
            data = generate_noise(data_preproc, multiplier=noise_power, noise_inputs=noise_inputs)
        else:
            data = torch.load(folder + 'data_{}_noise.pkl'.format(type_of_noise))

        ## Se generan nuevos batches
        if new_data:
            data, scaler_data, data_preproc, scaler_data_preproc = \
                self.make_batch_new_data_tanks(data=data, data_preproc=data_preproc, seqlen=self.seqlen, cpu=True, shuffle=shuffle)
        else:
            data, scaler_data, data_preproc, scaler_data_preproc = \
                self.make_batch_tanks(data=data, data_preproc=data_preproc, seqlen=self.seqlen, cpu=True, shuffle=shuffle)

        data_train = data[:int(data.shape[0] * ratio_train), :, :]
        data_train_preproc = data_preproc[:int(data.shape[0] * ratio_train), :, :]

        data_val = data[int(data.shape[0] * ratio_train):int(data.shape[0] * (ratio_train + ratio_val)), :, :]
        data_val_preproc = data_preproc[int(data.shape[0] * ratio_train):int(data.shape[0] * (ratio_train + ratio_val)), :, :]

        data_test = data[-int(data.shape[0] * ratio_test):, :, :]
        data_test_preproc = data_preproc[-int(data.shape[0] * ratio_test):, :, :]


        data_dict = {'train_data': data_train, 'val_data':data_val, 'test_data': data_test,
                     'train_data_preproc': data_train_preproc, 'val_data_preproc': data_val_preproc,
                     'test_data_preproc': data_test_preproc, 'scaler': scaler_data,
                     'scaler_preproc': scaler_data_preproc, 'signals': signals}


        return data_dict




if __name__ == '__main__':
    data_clean = np.zeros(shape=(10000, 6))
    #generate_noise(data_clean, multiplier=1.5, noise_inputs=False, plot=False)
    noise_power = 1
    noise_inputs = False
    noise = 'saltPepper'


    processor = DataProcessor(seqlen=60)
    data_tanks = processor.process_tanks(type_of_noise=noise, shuffle=False, noise_inputs=noise_inputs,
                                            noise_power=noise_power, folder='../Tanks_Data/No_Noised_Inputs/')
    scaler = data_tanks['scaler']
    scaler_preproc = data_tanks['scaler_preproc']
    test_data = scaler.inverse_transform(data_tanks['test_data'][:, -1, :].numpy())[:, 2:]
    test_data_preproc = scaler_preproc.inverse_transform(data_tanks['test_data_preproc'][:, -1, :].numpy())[:, 2:]
    rmse = np.sqrt(mean_squared_error(test_data, test_data_preproc))

    print('Noise Power: {} | RMSE: {}'.format(noise_power, rmse))