import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import time, os, torch
import numpy as np
from scipy.integrate import odeint
import datetime
from fcnQuadrupleTank_class import fcn_QuadrupleTank
from AE_buffer import *


if __name__ == "__main__":

    ## Instantiate AE
    folder='fcn_AE_models/'
    noise='saltPepper'
    name='uut_noise1'
    meta = load_metadata('{}{}/'.format(folder, name))
    processor = DataProcessor(seqlen=60)

    # data_dict2 no tiene shuffle
    data_dict2 = processor.process_tanks(noise_power=meta['noise_power'], noise_inputs=meta['noise_inputs'], shuffle=False,
                                                                folder='Tanks_Data/No_Noised_Inputs/', type_of_noise=noise, clean_data='data_fcn_clean.pkl')
    test_data = data_dict2['test_data']
    clean_data = data_dict2['test_data_preproc']
    print(test_data.shape)

    scaler = data_dict2['scaler']
    scaler_preproc = data_dict2['scaler_preproc']
    np_clean_data = clean_data.numpy()
    np_clean_data = scaler_preproc.inverse_transform(np_clean_data[:, 0, :])
    clean_data = torch.from_numpy(np_clean_data)
    print(clean_data.shape)

    AE = AutoEncoder(shape_data=test_data.shape[0])

    # Instanciate Plant
    x0=[40, 40, 40, 40]

    # ref -> reference for tanks 1 and 2
    ref = np.array([0, 0, 0, 0])
    plant = fcn_QuadrupleTank(x0=x0)
    # print(f'x0 = {plant.x}')

    first_run = 1
    sim_length = 40
    sim_points = 60000

    cnt = 0

    # Start time
    start_time = datetime.datetime.now()
    init_time = datetime.datetime.now()

    # Simulate the system with state feedback control
    run_sim = True
    while(run_sim):
        x = plant.x
        # print(f'clean x = {x}')

        ## Print x and u
        current_time = datetime.datetime.now()
        elapsed_time = (current_time - start_time).total_seconds()
        if elapsed_time >= 10:
            print(f'x = {x}')
            print(f'u = {plant.u}')
            print(f'calculated so far = {cnt}')
            start_time = datetime.datetime.now()
        
        ## Noise x (the function recieves a numpy array, a conversion from list to np.array has to be done and later reversed)
        x_np = np.array(x).reshape(1,4)
        # print(f'x_np = {x_np}')
        x_noised = AE.noise_datapoint(x_np, multiplier_white=1, multiplier_SP=1)
        x_noised = x_noised[0].tolist()
        # print(f'noised x = {x_noised}')
        
        # The point is concatenated to the input (not needed for buffer to work) and then reshaped
        # if 'u' in locals():
        #     point = np.concatenate((plant.u, x_noised), axis=None).reshape(1,6)
        # else:
        #     print(f'u is not local')
        #     point = np.concatenate((np.zeros((2,)), x_noised)).reshape(1,6)
        # print(f'point = {point}')
        point = np.concatenate((plant.u, x_noised), axis=None).reshape(1,6)
        
        # Fit data to scaling scheme (a np.array and a reshape is needed)
        # point_scaled = scaler.transform(np.array(point).reshape(-1,1))
        point_scaled = scaler.transform(np.array(point))

        # point_scaled = scaler.transform(np.array(point))
        point_scaled = np.concatenate(point_scaled, axis=None).reshape(1,6)
        
        ## Autoencoder
        AE.buffer(data_point=point_scaled)
        # Convert back
        # Y_pred_list = scaler.inverse_transform(np.concatenate(AE.Y_pred_list, axis=0))
        Y_pred_unscaled = AE.Y_pred_list[-1]
        Y_pred_list = scaler.inverse_transform(Y_pred_unscaled)
        # Y_pred_list = AE.Y_pred_list[-1]
        # print(f'Y_pred_list = {Y_pred_list}')
        # Select the last element of the buffer (tensor), not regarding the inputs (first two elements) and reshaping
        x_pred = Y_pred_list[-1][2:].reshape(4,)
        # print(f'x_pred = {x_pred}')

        ## Array concatenation index
        #   x_array -> clean states
        #   u_array -> clean inputs
        #   x_pred_array -> output of AE
        #   x_scaled_array -> noised states after scaling
        #   x_noised_array -> noised states (before AE and scaling) 
        if first_run:
            print(f'x0 = {x}')
            x_array = np.array(x)
            u_array = np.array(plant.u)
            x_pred_array = x_pred
            x_scaled_array = point_scaled
            x_noised_array = x_noised
            first_run = 0
        else:
            x_array = np.vstack((x_array, np.array(x)))
            u_array = np.vstack((u_array, np.array(plant.u)))
            x_pred_array = np.vstack((x_pred_array, x_pred))
            x_noised_array = np.vstack((x_noised_array, x_noised))
            x_scaled_array = np.vstack((x_scaled_array, point_scaled))

        ## Close loop dynamic
        plant.closed_loop(x_pred)

        # Break condition
        cnt += 1
        sim_time = (current_time - init_time).total_seconds()
        # if sim_time >= sim_length:
        if cnt > sim_points:
            print(f'Simulation compleated')
            run_sim = False

    # Check
    print(f'x.shape = {x_array.shape}')
    print(f'u.shape = {u_array.shape}')
    print(f'x_pred.shape = {x_pred_array.shape}')

    # Plotting the results
    # time_axis = np.linspace(0, sim_length, x_array.shape[0])
    time_axis = np.linspace(0, sim_points, x_array.shape[0])
    print(f'time_axis = {time_axis.shape}')

    # AE.scale()
    Y_pred_array = np.array([])
    for i in range(len(AE.Y_pred_list)):
        if Y_pred_array.shape[0] == 0:
            Y_pred_array = AE.Y_pred_list[i]
        else:
            Y_pred_array = np.vstack((Y_pred_array, AE.Y_pred_list[i]))
    print(f'Y_pred_array = {Y_pred_array.shape}')

## _________________________________________________________________________________________________________________________________

    # plt.figure(num=6)
    # for i in range(4):
    #     plt.subplot(2, 2, i+1)
    #     plt.plot(time_axis, Y_pred_array[:, i+2], label=f'x_AE_scaled_{i}')
    #     plt.title(f"Scaled state x_{i}")
    #     plt.legend()
    #     plt.grid()
    # plt.tight_layout()
    # plt.show()

    # plt.figure(num=1, figsize=(8, 6))
    # for i in range(4):
    #     # plt.plot(time_axis, Y_pred_array[:, i], label=f'x_AE{i}')
    #     plt.plot(time_axis, x_pred_array[:, i], label=f'x_AE{i}')
    # for i in range(4):    
    #     plt.plot(time_axis, x_array[:, i], label=f'State x_{i}')
    # plt.xlabel('Time')
    # plt.ylabel('State value')
    # plt.title('State Response with State Feedback Controller for AE')
    # plt.legend()
    # plt.grid()
    # plt.show()

    plt.figure(num=5)
    for i in range(4):
        plt.subplot(2, 2, i+1)
        plt.plot(time_axis, x_pred_array[:, i], label=f'x_AE_{i}')
        # plt.plot(time_axis, x_array[:, i], label=f'x_{i}') # Same as x_pred_array, but shifted one timestep
        plt.title(f"Filtered x_{i}")
        plt.legend()
        plt.grid()
    plt.tight_layout()
    plt.show()

    # plt.figure(3)
    # plt.plot(time_axis, x_scaled_array[:,2:], label='x_scaled')
    # plt.title('Scaled x')
    # plt.legend()
    # plt.grid()
    # plt.show()

    plt.figure(num=4)
    for i in range(4):
        plt.subplot(2, 2, i+1)
        plt.plot(time_axis, x_noised_array[:, i], label=f'x_noised_{i}')
        plt.plot(time_axis, x_array[:, i], label=f'x_{i}')
        plt.title(f"Clean and noised x_{i}")
        plt.legend()
        plt.grid()
    plt.tight_layout()
    plt.show()


    # plt.figure(num=2, figsize=(8, 6))
    # for i in range(2):
    #     plt.plot(time_axis, u_array[:, i], label=f'State u{i}')
    # plt.xlabel('Time')
    # plt.ylabel('Input')
    # plt.title('State Response with State Feedback Controller')
    # plt.legend()
    # plt.grid()
    # plt.show()

    plt.figure(num=7)
    for i in range(4):
        plt.subplot(2, 2, i+1)
        plt.plot(x_pred_array[:, i], label=f'x_noisy')
        plt.plot(x_noised_array[:, i], label=f'x_AE')
        plt.plot(x_array[:, i], label=f'x_clean')
        plt.title(f"State x_{i}")
        plt.legend()
        plt.grid()
    plt.tight_layout()
    plt.show()

    ## Saving
    # Create directory
    if not os.path.exists('CL_AE_data/'):
        os.makedirs('CL_AE_data/')

    torch.save(x_array, 'CL_AE_data/x_array.pkl')
    torch.save(u_array, 'CL_AE_data/u_array.pkl')
    torch.save(x_pred_array, 'CL_AE_data/x_pred_array.pkl')
    torch.save(x_noised_array, 'CL_AE_data/x_noised_array.pkl')
    torch.save(x_scaled_array, 'CL_AE_data/x_scaled_array.pkl')
    print('Saving process completed')