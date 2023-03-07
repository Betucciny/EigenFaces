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
    print(correctas / test_eigen.shape[1] * 100)


if __name__ == '__main__':
    main()
