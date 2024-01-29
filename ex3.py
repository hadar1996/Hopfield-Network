import matplotlib.pyplot as plt
import numpy as np
import random

value = 0


# Read the input data
def read_data():
    file = open("assbio3.txt", 'r')
    data = []
    temp = []
    for line in file:
        if line.rstrip() == "":
            temp = np.array(temp)
            temp = np.where(temp == 0, value, temp)
            data.append(temp)
            temp = []
        else:
            temp.append([int(bit) for bit in line.rstrip()])
    temp = np.array(temp)
    temp = np.where(temp == 0, value, temp)
    data.append(temp)
    file.close()
    return data


# Print the matrix image
def show_matrix(matrix):
    plt.imshow(matrix)
    plt.show()


def randomize(matrix_list):
    for i in range(len(matrix_list)):
        for j in range(10):
            num = random.randint(0, 99)
            if matrix_list[j][num] == 0:
                matrix_list[j][num] = 1
            else:
                matrix_list[j][num] = 0
    return matrix_list


def convert_to_matrix(matrix):
    new_matrix = matrix.reshape(10, 10)
    return new_matrix


# Create the matrix database
def create():
    data = read_data()
    all_matrices = []
    current_matrix = []
    for row in data:
        current_matrix.append(row.reshape(100))
        if len(current_matrix) == 10:
            all_matrices.append(np.array(current_matrix))
            current_matrix = []
    return all_matrices


def train_network(matrix):
    values = []
    for i in range(100):
        line = []
        for j in range(100):
            grade = 0
            for k in range(10):
                if matrix[k][i] == matrix[k][j]:
                    grade += 1
                else:
                    grade -= 1
            line.append(grade)
        values.append(line)
    return np.array(values)


def train_all(matrix_list):
    trained = []
    for matrix in matrix_list:
        trained.append(train_network(matrix))
    return trained


def recover(matrix, trained_matrix):
    pre_matrix = np.copy(matrix)
    after_matrix = np.copy(matrix)

    bit_order = np.arange(100)
    np.random.shuffle(bit_order)
    flag = True
    while flag:
        flag = False
        for bit in range(100):
            sum = 0
            for i in range(100):
                if i != bit_order[bit]:
                    sum = sum + (after_matrix[i] * trained_matrix[bit_order[bit]][i])
            memory = after_matrix[bit_order[bit]]
            if sum >= 0:
                after_matrix[bit_order[bit]] = 1
            else:
                after_matrix[bit_order[bit]] = value
            if memory != after_matrix[bit_order[bit]]:
                flag = True
    show_matrix(pre_matrix)
    show_matrix(after_matrix)
    return after_matrix



scores = []
data_base = create()
random_data = randomize(data_base)
trained_matrices = train_all(random_data)
print(trained_matrices)

"""
nums_scores = []
    for num_lern in range(1, 11, 1):
        t_of_all = np.zeros((100, 100))
        for i in range(num_lern):
            t_of_all = t_of_all + matrices_t[i]
        t_of_all = t_of_all / (num_lern + 1)

        sum_n_avg = 0
        for i in range(10):
            recon_matrix = reconstruction(mutated_data[i], t_of_all)
            sum_n_avg = sum_n_avg + np.linalg.norm(data[0].reshape(100) - recon_matrix)
            sum_n_avg = sum_n_avg / 10
        nums_scores.append(sum_n_avg)
    scores.append(nums_scores)
"""