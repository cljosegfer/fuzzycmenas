import numpy as np
from cmenas import cmenas
from manipulacao import photo_open, pick_pixels, coloring
from norm import normalize, denormalize
import time

#leitura de dados
file_seq = ['photo001.jpg', 'photo002.jpg', 'photo003.jpg', 'photo004.jpg', 
            'photo005.jpg', 'photo006.jpg', 'photo007.jpg', 'photo008.jpg', 
            'photo009.jpg', 'photo010.jpg', 'photo011.png']
duration = 0

for image in range(len(file_seq)):
    start_time = time.time()
    filename = file_seq[image]
    path = 'fig/' + filename
    photo = photo_open(filename = 'ImagensTeste/' + filename)
    pixels = pick_pixels(photo = photo)
    #normalizacao
    data, minimum, maximum = normalize(pixels)

    #cmeans
    k = 15
    model = cmenas(k = k)
    MAX = 25
    tol = 1e-2
    model.train(data = data, MAX = MAX, tol = tol, log = False)

    #get layers (boolean) and centers
    pertinencia = np.zeros(model.U.shape)
    pertinencia[range(len(model.U)), model.U.argmax(1)] = 1 #max column is True
    pertinencia = np.array(pertinencia, dtype = bool)
    centros = model.C

    #denormalizacao
    centros = denormalize(data = centros, m = minimum, M = maximum)

    #geracao
    gen_photo = coloring(photo = photo, labels = pertinencia, centers = centros)
    gen_photo.save(path)
    deltat = time.time() - start_time
    duration += deltat
    print(filename, 'duration:', deltat)

print('finish', duration)