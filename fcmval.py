import numpy as np
from scipy.io import loadmat
from matplotlib import pyplot as plt
from cmenas import cmenas
from norm import normalize, denormalize

#leitura de dados
data = np.array(loadmat('fcm_dataset.mat')['x'])
#normalize data, should we do it??
data, minimum, maximum = normalize(data = data)

#cmeans
k = 4
cmenas = cmenas(k = k)
MAX = 15
tol = 0.01
cmenas.train(data = data, MAX = MAX, tol = tol)
data = denormalize(data = data, m = minimum, M = maximum)
centros = denormalize(data = cmenas.centros, m = minimum, M = maximum)
print(centros)

#plot
plt.scatter(x = data[:, 0], y = data[:, 1])
plt.scatter(x = centros[:, 0], y = centros[:, 1])
plt.show()
# plt.scatter(x = data[:, 0], y = data[:, 1], color = cmenas.U)