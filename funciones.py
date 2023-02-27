import numpy as np
import cv2


# Calcular la matriz de covarianza
def covariance_matrix(x):
    centered_x = x - np.mean(x, axis=0, keepdims=True)
    return np.matmul(centered_x, centered_x.T) / (x.shape[1])


# Vectorizar imagen
def vectorimg(img):
    vec = []
    for fila in img:
        vec += fila.tolist()
    return vec


# Importing the dataset
def faces():
    matrizcompleta = []
    for i in range(1, 41):
        folder = 's' + str(i)
        for j in range(1, 6):
            filename = 'dataset/' + folder + '/' + str(j) + '.pgm'
            img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
            img = np.array(cv2.resize(img, (64, 64)))
            vec = vectorimg(img)
            matrizcompleta.append(vec)
    return np.array(matrizcompleta)


# Calcular la matriz de eigenfaces
def eigenmatrix(datos, k):
    cov = covariance_matrix(datos)
    autovalores, eigenvectors = np.linalg.eig(cov)
    newautovectores = []
    for i in range(eigenvectors.shape[1]):
        eigenvector = eigenvectors[:, i]
        temp = datos.T @ eigenvector
        norm = np.linalg.norm(temp)
        temp = temp / norm
        newautovectores.append(temp)

    autovectores = np.array(newautovectores)
    eigen = [(i, j) for i, j in zip(autovalores, autovectores)]
    eigen.sort(key=lambda x: x[0], reverse=True)
    eigenmat = np.array([i[1] for j, i in enumerate(eigen) if j < k])
    print(eigenmat)
    return eigenmat, cov


# Descomponer imagen en sus componentes principales
def deconstruct(img,  k=200,  eigenmat=np.loadtxt('eigenmat.txt')):
    eigenmat = eigenmat[0:k, :]
    vec, mean = veccent(img)
    dec = eigenmat @ vec
    return dec, mean


# Reconstruir imagen a partir de sus componentes principales
def reconstruct(prueba, mean, k=200, eigenmat=np.loadtxt('eigenmat.txt')):
    eigenmat = eigenmat[0:k, :]
    normal = eigenmat.T @ prueba + mean
    return normal


# Vectorizar imagen centrada
def veccent(img):
    vec = vectorimg(img)
    mean = np.mean(vec, axis=0, keepdims=True)
    return vec - mean, mean


# Guardar matriz en archivo
def guardar_matriz(matriz, archivo):
    np.savetxt(archivo, matriz)



