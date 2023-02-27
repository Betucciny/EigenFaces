from funciones import *


# Main
def main():
    datos = faces()
    eigenmat, cov = eigenmatrix(datos, 200)
    guardar_matriz(eigenmat, 'eigenmat.txt')
    guardar_matriz(cov, 'cov.txt')


if __name__ == '__main__':
    main()