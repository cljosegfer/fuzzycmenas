import numpy as np
from cmenas import cmenas
from manipulacao import photo_open, pick_pixels, coloring
from norm import normalize, denormalize

#leitura de dados
filename = 'photo005.jpg'
photo = photo_open(filename = 'ImagensTeste/' + filename)
pixels = pick_pixels(photo = photo)
#normalizacao
# data, minimum, maximum = normalize(pixels)
data = pixels

#cmeans
k = 6
cmenas = cmenas(k = k)
MAX = 5
tol = 0.01
cmenas.train(data = data, MAX = MAX, tol = tol)
pertinencia = np.zeros(cmenas.U.shape)
pertinencia[range(len(cmenas.U)), cmenas.U.argmax(1)] = 1
pertinencia = np.array(pertinencia, dtype = bool)
centros = cmenas.centros
# centros = denormalize(data = cmenas.centros, m = minimum, M = maximum)

#geracao
gen_photo = coloring(photo = photo, labels = pertinencia, centers = centros)
gen_photo.save('fig/' + filename + '.png')