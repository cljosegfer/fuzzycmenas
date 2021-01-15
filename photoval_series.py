import numpy as np
from cmenas import cmenas
from manipulacao import photo_open, pick_pixels, coloring
from norm import normalize, denormalize

file_seq = ['photo001.jpg', 'photo002.jpg', 'photo003.jpg', 'photo004.jpg', 
            'photo005.jpg', 'photo006.jpg', 'photo007.jpg', 'photo008.jpg', 
            'photo009.jpg', 'photo010.jpg', 'photo011.png']

for image in range(len(file_seq)):
    #leitura de dados
    filename = file_seq[image]
    path = filename[:-4] + '/'
    print(filename)
    photo = photo_open(filename = 'ImagensTeste/' + filename)
    pixels = pick_pixels(photo = photo)
    #normalizacao
    data, minimum, maximum = normalize(pixels)
    
    N = 20
    k_seq = [5, 10, 15, 20]
    for idx in range(len(k_seq)):
        loss = 10e3
        loss_vector = np.zeros([N, 1])
        for i in range(N):
            # print(i)
            #cmeans
            k = k_seq[idx]
            model = cmenas(k = k)
            MAX = 25
            tol = 0.0001
            model.train(data = data, MAX = MAX, tol = tol, log = False)
            loss_vector[i] = model.J
            #get layers (boolean) and centers
            if (model.J < loss):
                loss = model.J
                pertinencia = np.zeros(model.U.shape)
                pertinencia[range(len(model.U)), model.U.argmax(1)] = 1
                pertinencia = np.array(pertinencia, dtype = bool)
                centros = model.C
        
        # print(loss_vector.argmin(), loss_vector.min())
        #denormalizacao
        centros = denormalize(data = centros, m = minimum, M = maximum)
        
        #geracao
        gen_photo = coloring(photo = photo, labels = pertinencia, 
                             centers = centros)
        gen_photo.save('fig/best/' + path + 'k' + str(k) + filename)
        print('printed in', 'fig/best/' + path + 'k' + str(k) + filename)