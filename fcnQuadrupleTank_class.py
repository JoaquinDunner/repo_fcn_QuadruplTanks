import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import time, os, torch
import numpy as np
from scipy.integrate import odeint
import datetime

class fcn_QuadrupleTank():
    def __init__(self, x0):
        self.x0 = x0

        # Parameters
        self.x = self.x0
        self.u = np.zeros((2,))
        self.Ts = 0.01
        self.current_time = 0
        self.voltmax = 10

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
        # self.K = np.zeros((2,4))    # Zero input check (open loop)

        # Closed-loop dynamics (A - BK)
        self.CL_dyn = self.A - (self.B @ self.K)

        # Initial input (random)
        self.u = np.zeros((2,))

    # Integración en "tiempo real"
    def closed_loop(self, x_in):
        # Calculate control input u = -Kx
        self.u = -self.K @ x_in

        # Update state using A - BK
        # self.x = self.x + self.Ts * (self.CL_dyn @ self.x)
        # self.x = self.x + self.Ts * (self.A @ self.x + self.B @ self.u)
        self.x = self.x + self.Ts * (self.CL_dyn @ x_in)

        # Increment time
        self.current_time += self.Ts

