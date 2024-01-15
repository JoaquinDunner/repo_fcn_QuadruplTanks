import matplotlib.pyplot as plt
import torch

data = torch.load("data_clean_lin.pkl")
print(data.shape)

plt.figure(1)
plt.plot(data[:3000,2:])
plt.title(f'My Data Outputs')
plt.figure(2)
plt.plot(data[:,:2])
plt.title(f'My Data Inputs')
plt.show()