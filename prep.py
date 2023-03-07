from funciones import *


# Main
def main():
    datos = faces_train()
    eigenmat, cov = eigenmatrix(datos, 190)
    eigen_faces, mean = eigenfaces(eigenmat, faces_train(), 190)
    guardar_matriz(eigenmat, 'eigenmat.txt')
    guardar_matriz(eigen_faces, 'eigenfaces.txt')
    guardar_matriz(mean, 'mean.txt')


if __name__ == '__main__':
    main()
