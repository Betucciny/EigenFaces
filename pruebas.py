from funciones import *


def main():
    img = np.array(cv2.resize((cv2.imread('dataset/s6/2.pgm', cv2.IMREAD_GRAYSCALE)), (64, 64)))
    cv2.imwrite('original.png', img.reshape(64, 64))

    k = 50
    deconstruida, mean = deconstruct(img, k)
    reconstruida = reconstruct(deconstruida, mean, k)

    cv2.imwrite('reconstruida.png', reconstruida.reshape(64, 64))


if __name__ == '__main__':
    main()
