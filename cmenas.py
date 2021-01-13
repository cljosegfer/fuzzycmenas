import numpy as np
import time

class cmenas:
    def __init__(self, k):
        # k : numero de clusters
        # centros : matriz q guarda o espaco dos centros
        # U : matriz q mapeia as amostras nos grupos
        self.k = k
        self.centros = np.array([])
        self.U = np.array([])
    
    def train(self, data, MAX, tol):
        # data : conjunto de treinamento
        # MAX : numero maximo de epocas
        # tol : tolerancia da funcao de custo
        num_amostras, num_atributos = data.shape
        #STEP 1: initialize U and normalize / initialize centers
        # self.U = np.random.uniform(size = (num_amostras, self.k))
        # self.U = self.U / self.U.sum(axis = 1)[:, np.newaxis]
        index = np.random.choice(a = num_amostras, size = self.k, 
                                 replace = False)
        self.centros = data[index, :]
        #add noisy
        self.centros += np.random.normal(size = self.centros.shape) * 0.001
        #iteration
        stop = False
        ite = 0
        while (stop == False):
            start_time = time.time()
            #STEP 4: compute new U
            self.U = self.comp_memb(data = data, c = self.k, 
                                    centros = self.centros, m = 2)
            #STEP 3: now, calculate cost function
            J = self.comp_cost(data = data, U = self.U, c = self.k, 
                               centros = self.centros, m = 2)
            #STEP 2: calculate centers, m is often 2 so..
            self.centros = self.calc_centros(data = data, U = self.U, 
                                             k = self.k, m = 2)
            #condicao de parada
            ite += 1
            print(ite, J, time.time() - start_time)
            if (ite >= MAX or tol > J): stop = True
    
    def calc_centros(self, data, U, k, m):
        n, num_atributos = data.shape
        centros = np.zeros([k, num_atributos])
        #equation 15.8
        for i in range(k):
            num = 0
            den = 0
            for j in range(n):
                u = U[j, i]
                x = data[j, :]
                num += (u ** m) * x
                den += u ** m
            centros[i, :] = num / den
        return centros
    
    def comp_cost(self, data, U, c, centros, m):
        n, num_atributos = data.shape
        #equation 15.6
        J = 0
        for i in range(c):
            Ji = 0
            for j in range(n):
                u = U[j, i]
                x = data[j, :]
                ci = centros[i, :]
                d = np.linalg.norm(ci - x)
                Ji += (u ** m) * d
            J += Ji
        #i think this makes sense, idk
        J = J / n / c
        return J
    
    def comp_memb(self, data, c, centros, m):
        n, num_atributos = data.shape
        #equation 15.9
        U = np.zeros([n, c])
        for i in range(c):
            for j in range(n):
                num = 1
                den = 0
                for k in range(c):
                    x = data[j, :]
                    ci = centros[i, :]
                    ck = centros[k, :]
                    dij = np.linalg.norm(ci - x)
                    dkj = np.linalg.norm(ck - x)
                    den += (dij / dkj) ** (1 / (m - 1))
                U[j, i] = num / den
        return U