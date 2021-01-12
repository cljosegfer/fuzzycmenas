import numpy as np
from PIL import Image

def photo_open(filename, rescale = 1):
    photo = Image.open(filename)
    photo = photo.convert('RGB')
    photo = photo.resize((int(photo.size[0] / rescale),
                          int(photo.size[1] / rescale)), Image.ANTIALIAS)
    return photo

def pick_pixels(photo):
    n , m = photo.size
    ibagem = []     #eu quero ibagens
    pixels = photo.load()
    for i in range(n):
        for j in range(m):
            ibagem.append(list(pixels[i, j]))
    return np.array(ibagem)

def coloring(photo, labels, centers, rescale = 1):
    n, m = photo.size
    pixels = photo.load()
    for i in range(n):
        for j in range(m):
            # numb = [int(number) for number in centers[labels[i * m + j]]]
            numb = [number for number in centers[labels[i * m + j]].astype(int)]
            # pixels[i, j] = tuple(numb)
            pixels[i, j] = tuple(map(tuple, numb))
    photo = photo.resize((int(photo.size[0] * rescale), 
                          int(photo.size[1] * rescale)), Image.ANTIALIAS)
    return photo