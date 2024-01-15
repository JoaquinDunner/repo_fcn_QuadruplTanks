import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import time
import sys
import random
import threading
from scipy.io import loadmat, savemat
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
import pandas as pd
import torch, os


class fcn_QuadrupleTank():
    def __init__(self, x0):
        self.x0 = x0

        # Parameters
        self.x = self.x0
        self.u = np.zeros((2,))
        self.Ts = 0.01
        self.current_time = 0

        # State-space system
        self.A = np.array([ [-0.0089, 0,          0.0089,     0       ],
                            [0,       -0.0062,    0,          0.0101  ],
                            [0,       0,          -0.0089,    0       ],
                            [0,       0,          0,          -0.0062 ] ])  
        self.B = np.array([ [0.0833,  0       ],
                            [0,       0.0624  ],
                            [0,       0.0476  ],
                            [0.0312,  0       ] ])   
        self.C = np.array([ [1, 0,  0,  0],
                            [0, 1,  0,  0]    ])  
        self.D = np.zeros((2, 2)) 

        # Use LQR to design the state feedback gain matrix K
        self.K = np.array([     [0.7844,    0.1129,   -0.0768,    0.5117],
                                [0.0557,    0.7388,    0.5409,   -0.0397]   ])

        # Closed-loop dynamics (A - BK)
        self.CL_dyn = self.A - self.B @ self.K

    # Open loop dynamics: input is u (2x1) and output is x (4x1)
    def open_loop(self, u):
        # Calculate the deviation from equilibrium
        dx = self.A @ self.x + self.B @ u

        # Update of the state
        self.x = self.x + self.Ts * dx

        # Increment time
        self.current_time += self.Ts

    # Closed loop dynamics: states and inputs are saved iteratively in x and u, respectively
    def closed_loop(self):
        # Calculate control input u = -Kx
        self.u = -self.K @ self.x

        # Update state using A - BK
        self.x = self.x + self.Ts * (self.CL_dyn @ self.x)

        # Increment time
        self.current_time += self.Ts


def generate_pattern(n_seqs=500, seqlen=60):
    u1_seqs = []
    u2_seqs = []
    ## Generar vector con n_seq secuencias de largo seqlen, en que cada una contiene un número aleatorio >= 0.05
    for n in range(n_seqs):
        u1_val = max(random.random(), 0.05)
        u2_val = max(random.random(), 0.05)

        u1_seqs.append([u1_val for i in range(seqlen)])
        u2_seqs.append([u2_val for i in range(seqlen)])

    ## Estirar secuencias para que quede una lista de largo n_seq * seqlen
    u1_seqs = np.concatenate(u1_seqs).reshape(-1, 1)
    u2_seqs = np.concatenate(u2_seqs).reshape(-1, 1)

    ## Matriz de secuencias: n_seq * seqlen x 2
    u_seqs = np.concatenate([u1_seqs, u2_seqs], axis=1)
    ## Sumar ruido gaussiano con media = 0 y std = 0.01
    u_seqs = u_seqs + np.random.normal(loc=np.zeros_like(u_seqs), scale=np.ones_like(u_seqs)*0.01)
    ## Hace una ventana de tamaño window y la aplica la función mean(). Los primeros 19 valores son NaN, luego empieza la media
    u_seqs = np.array(pd.DataFrame(u_seqs).rolling(window=20, min_periods=0, win_type='hamming').mean())
    ## Sumar ruido gaussiano con media = 0 y std = 0.0015
    u_seqs = u_seqs + np.random.normal(loc=np.zeros_like(u_seqs), scale=np.ones_like(u_seqs)*0.0015)
    ## Comentado de antes
    #u_seqs = savgol_filter(u_seqs, window_length=21, polyorder=3, axis=0)

    return u_seqs

if __name__ == '__main__':
    series = []
    sistema = fcn_QuadrupleTank(x0=[40,40,40,40])
    sistema.time_scaling = 1 # Para el tiempo

    ## Mostrar hasta esa muestra
    upto = 20000

    ## Input generation
    inputs = generate_pattern(n_seqs=3000, seqlen=60)
    ## Inputs must be rescaled to make sense: in actuality, the voltage ranged between 0-10 V
    inputs *= 10
    plt.figure(1)
    plt.plot(inputs[:upto, :])
    plt.show()
    print(f'Pattern shape = {inputs.shape}')

    #noise = np.random.normal(loc=0, scale=0.05, size=inputs.shape)
    #inputs = inputs + noise
    #print(inputs.shape)

    for i in range(len(inputs)):
        ## Sequence starts with NaN, so we skip them
        if not np.isnan(np.array(inputs[i, :])).any():
            u_input = list(inputs[i, :])  # Rando input generated
            # u_input = [0, 0]                # Zero input check
            sistema.open_loop(u=u_input)
            series.append(sistema.x)

    series = np.array(series)
    print(f'series.shape = {series.shape}')

    ## Truncate the input series to eliminate the NaN values
    inputs = inputs[-series.shape[0]:, :]
    print(f'inputs.shape = {inputs.shape}')

    upto = 20*10**3
    # plt.figure(2)
    # #plt.title('Alturas de los tanques [cm]')
    # plt.subplot(2, 1, 1)
    # plt.plot(series[:upto, 0], label='Tanque1')
    # plt.plot(series[:upto, 1], label='Tanque2')
    # plt.plot(series[:upto, 2], label='Tanque3')
    # plt.plot(series[:upto, 3], label='Tanque4')
    # plt.legend()
    # plt.grid()

    # #plt.figure(3)
    # #plt.title('Inputs')
    # plt.subplot(2, 1, 2)
    # plt.plot(inputs[:upto, 0], label='Input1')
    # plt.plot(inputs[:upto, 1], label='Input2')
    # plt.grid()
    # plt.legend()

    # plt.show()

    plt.figure(3)
    plt.title('Estados generados')
    plt.plot(series[:, 0], label='x_0')
    plt.plot(series[:, 1], label='x_1')
    plt.plot(series[:, 2], label='x_2')
    plt.plot(series[:, 3], label='x_3')
    plt.legend()
    plt.grid()
    plt.show()

    ## Crear directorio
    if not os.path.exists('No_Noised_Inputs/'):
            os.makedirs('No_Noised_Inputs/')

    ## Guardar data
    data_to_save = np.concatenate([inputs, series], axis=1)
    print(data_to_save.shape)

    ## Chequeo de NaN values
    has_nan = np.isnan(np.array(data_to_save)).any()
    if has_nan:
        print("The matrix contains NaN values.")
        data_to_save = np.array([[0 if np.isnan(element) else element for element in row] for row in data_to_save])

    has_nan = np.isnan(np.array(data_to_save)).any()
    if has_nan:
        print("The matrix STILL contains NaN values.")
    else:
        print("NaN values were deleted.")
    print(data_to_save.shape)
    np.save('No_Noised_Inputs/data_fcn_clean.npy', data_to_save)
    #np.save('data_noise.npy', np.concatenate([inputs_noise, series_noise], axis=1))
    torch.save(data_to_save, 'No_Noised_Inputs/data_fcn_clean.pkl')
    print(f'Saved \'data_to_save\'')


