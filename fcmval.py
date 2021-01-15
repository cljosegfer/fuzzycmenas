import numpy as np
from scipy.io import loadmat
from matplotlib import pyplot as plt
from cmenas import cmenas
from norm import normalize, denormalize

#leitura de dados
data = np.array(loadmat('fcm_dataset.mat')['x'])
#normalize data
data, minimum, maximum = normalize(data = data)

#cmeans
k = 4
model = cmenas(k = k)
MAX = 15
tol = 1e-2
model.train(data = data, MAX = MAX, tol = tol)
centros = model.C

#denormalizacao
data = denormalize(data = data, m = minimum, M = maximum)
centros = denormalize(data = centros, m = minimum, M = maximum)
# print(centros)

#plot
plt.scatter(x = data[:, 0], y = data[:, 1])
plt.scatter(x = centros[:, 0], y = centros[:, 1])
plt.show()
# plt.scatter(x = data[:, 0], y = data[:, 1], color = model.U)