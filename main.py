import matplotlib.pyplot as plt
import numpy as np
import random


N = 10
nb_patterns = 1


def zero():
    with open('zero_matrix.txt') as matrices_file:
        matrices_buffer_list = matrices_file.buffer.read().decode().split("\r\n")
    a = [[int(ch) for ch in element] for element in [matrix for matrix in matrices_buffer_list]]
    matrices = np.array(a)
    return matrices


def show_matrix(num_list):
    plt.imshow(num_list)
    plt.show()


def convert_to_matrix(matrix):
    new_matrix = matrix.reshape(N, N)
    return new_matrix


def random_line(matrix):
    for i in matrix:
        num = random.randint(0, 9)
        if i[num] == 0:
            i[num] = 1
        else:
            i[num] = 0
    return matrix


if __name__ == '__main__':
    zero_matrix = zero()
    w = []
    w.append(zero_matrix[0])
    w.append(one_matrix[0])

    print(zero_matrix[0])
    test = convert_to_matrix(zero_matrix[0])
    print("__________________________")
    #show_matrix(test)
    test_rand = random_line(test)
    #show_matrix(test_rand)