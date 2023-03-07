from funciones import *


def main():
    k = 190
    eigenmat = np.loadtxt('eigenmat.txt')
    train_eigen = np.loadtxt('eigenfaces.txt')

    test_images = faces_test()
    test_eigen, test_mean = eigenfaces(eigenmat, test_images, k)

    correctas = 0
    for i in range(test_eigen.shape[1]):
        test = test_eigen[:, i]
        distancia = []
        for j in range(train_eigen.shape[1]):
            train = train_eigen[:, j]
            distancia.append(np.linalg.norm(train - test))
        index = np.argmin(distancia)
        clase_ideal = i // 5 + 1
        clase_test = index // 5 + 1
        if clase_ideal == clase_test:
            correctas += 1
    porcentaje_correctas = correctas / test_eigen.shape[1] * 100
    print(f'Porcentaje de aciertos: {porcentaje_correctas}%')

    num_imagen = 1
    imagen_original = test_images[num_imagen, :]
    imagen_original = imagen_original.reshape(92, 112)

    imagen_reconstruida = test_eigen[:, num_imagen]
    imagen_reconstruida = reconstruct(imagen_reconstruida, test_mean[num_imagen], k, eigenmat)
    print(imagen_reconstruida.shape)

    imagen_reconstruida = imagen_reconstruida.reshape(92, 112)

    cv2.imwrite('original_test.png', imagen_original)
    cv2.imwrite('reconstruida_test.png', imagen_reconstruida)




if __name__ == '__main__':
    main()
