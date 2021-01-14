import numpy as np
from cmenas import cmenas
from manipulacao import photo_open, pick_pixels, coloring
from norm import normalize, denormalize

#leitura de dados
filename = 'photo005.jpg'
photo = photo_open(filename = 'ImagensTeste/' + filename)
pixels = pick_pixels(photo = photo)
#normalizacao
data, minimum, maximum = normalize(pixels)

#cmeans
k = 20
model = cmenas(k = k)
MAX = 25
tol = 0.0001
model.train(data = data, MAX = MAX, tol = tol)

#get layers (boolean) and centers
pertinencia = np.zeros(model.U.shape)
pertinencia[range(len(model.U)), model.U.argmax(1)] = 1 #max column is True
pertinencia = np.array(pertinencia, dtype = bool)
centros = model.centros

#denormalizacao
centros = denormalize(data = model.centros, m = minimum, M = maximum)

#geracao
gen_photo = coloring(photo = photo, labels = pertinencia, centers = centros)
gen_photo.save('fig/' + filename)