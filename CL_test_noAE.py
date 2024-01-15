import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import time, os, torch
import numpy as np
from scipy.integrate import odeint
import datetime
from fcnQuadrupleTank_class import fcn_QuadrupleTank


if __name__ == "__main__":

    # Start time
    start_time = datetime.datetime.now()
    init_time = datetime.datetime.now()

    # Instanciate Plant
    x0=[40, 40, 40, 40]

    # ref -> reference for tanks 1 and 2
    ref = np.array([0, 0, 0, 0])
    plant = fcn_QuadrupleTank(x0=x0)

    print(f'x0 = {plant.x}')

    first_run = 1
    sim_points = 1000
    sim_length = 10
    cnt = 0

    # Simulate the system with state feedback control
    run_sim = True
    while(run_sim):
        x = plant.closed_loop(plant.x)
        # print(x)

        ## Print x
        current_time = datetime.datetime.now()
        elapsed_time = (current_time - start_time).total_seconds()
        if elapsed_time >= 1:
            print(f'x = {plant.x}')
            print(f'u = {plant.u}')
            start_time = datetime.datetime.now()

        # Array concatenation
        if first_run:
            x_array = np.array(plant.x)
            u_array = np.array(plant.u)
            first_run = 0
        else:
            x_array = np.vstack((x_array, np.array(plant.x)))
            u_array = np.vstack((u_array, np.array(plant.u)))

        # Break condition
        cnt += 1
        sim_time = (current_time - init_time).total_seconds()
        if sim_time >= sim_length:
        # if cnt > sim_points:
            print(f'Simulation compleated')
            run_sim = False

    # Check
    print(f'x.shape = {x_array.shape}')
    print(f'u.shape = {u_array.shape}')

    # Plotting the results
    time_axis = np.linspace(0, sim_length, x_array.shape[0])
    print(f'time_axis = {time_axis.shape}')

    plt.figure(num=1, figsize=(8, 6))
    for i in range(4):
        plt.plot(time_axis, x_array[:, i], label=f'State x_{i}')
    plt.xlabel('Time')
    plt.ylabel('State value')
    plt.title('State Response with State Feedback Controller')
    plt.legend()
    plt.grid()
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

    ## Crear directorio
    if not os.path.exists('CL_No_Noised_Inputs/'):
        os.makedirs('CL_No_Noised_Inputs/')

    ## Guardar data
    data_to_save = np.concatenate([u_array, x_array], axis=1)
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

    # np.save('CL_No_Noised_Inputs/data_clean_CL.npy', data_to_save)
    #np.save('data_noise.npy', np.concatenate([inputs_noise, series_noise], axis=1))
    # torch.save(data_to_save, 'CL_No_Noised_Inputs/data_clean_CL.pkl')
    # print('Saving process completed')
