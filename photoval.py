import numpy as np
from cmenas import cmenas
from manipulacao import photo_open, pick_pixels, coloring
from norm import normalize, denormalize

#leitura de dados
# k_seq = [9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9]
file_seq = ['photo001.jpg', 'photo002.jpg', 'photo003.jpg', 'photo004.jpg',
            'photo005.jpg', 'photo006.jpg', 'photo007.jpg', 'photo008.jpg',
            'photo009.jpg', 'photo010.jpg', 'photo011.png']
for ind in range(len(k_seq)):
    # filename = 'photo005.jpg'
    filename = file_seq[ind]
    photo = photo_open(filename = 'ImagensTeste/' + filename)
    pixels = pick_pixels(photo = photo)
    #normalizacao
    data, minimum, maximum = normalize(pixels)
    # data = pixels
    
    #cmeans
    k = 5
    # k = k_seq[ind]
    model = cmenas(k = k)
    MAX = 5
    tol = 0.001
    print(filename)
    model.train(data = data, MAX = MAX, tol = tol)
    pertinencia = np.zeros(model.U.shape)
    pertinencia[range(len(model.U)), model.U.argmax(1)] = 1
    pertinencia = np.array(pertinencia, dtype = bool)
    centros = model.centros
    # print(centros.astype(int))
    centros = denormalize(data = model.centros, m = minimum, M = maximum)
    
    #geracao
    gen_photo = coloring(photo = photo, labels = pertinencia, centers = centros)
    gen_photo.save('fig/' + filename)