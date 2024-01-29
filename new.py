import numpy as np
from matplotlib import pyplot as plt

one_opposite = -1


def print_matrix(matrix):
    class Color:
        BLACK = '\033[30m'
        RED = '\033[31m'
        RESET = '\033[0m'
    for i in matrix:
        for j in i:
            if j == one_opposite:
                print(Color.BLACK + "██", end='')
            else:
                print(Color.RED + "██", end='')
        print(Color.RESET)
    print()


def modificate_matrix(matrix, rate):
    matrix_list = []
    for _ in range(10):
        index_array = np.arange(matrix.size)
        np.random.shuffle(index_array)
        index_array = index_array[:rate]

        locations_i = []
        locations_j = []
        for index in index_array:
            locations_i.append(int(index / 10))
            locations_j.append(index % 10)

        mutated_matrix = np.copy(matrix)
        for m in range(rate):
            if mutated_matrix[locations_i[m]][locations_j[m]] == 1:
                mutated_matrix[locations_i[m]][locations_j[m]] = one_opposite
            else:
                mutated_matrix[locations_i[m]][locations_j[m]] = 1
        matrix_list.append(mutated_matrix)
    return matrix_list


def make_T(matrix):
    scores = []
    for cul1 in range(100):
        line_score = []
        for cul2 in range(100):
            score = 0
            for index in range(10):
                if matrix[index][cul1] == matrix[index][cul2]:
                    score = score + 1
                else:
                    score = score - 1
            line_score.append(score)
        scores.append(line_score)
    return np.array(scores)


def reconstruction(matrix, t_mat):
    before_matrix = np.copy(matrix)
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
                    sum = sum + (after_matrix[i] * t_mat[bit_order[bit]][i])
            memory = after_matrix[bit_order[bit]]
            if sum >= 0:
                after_matrix[bit_order[bit]] = 1
            else:
                after_matrix[bit_order[bit]] = one_opposite
            if memory != after_matrix[bit_order[bit]]:
                flag = True
    return after_matrix
    # print_matrix(before_matrix.reshape(10, 10))
    # print_matrix(after_matrix.reshape(10, 10))


def modificate_matrix(matrix, rate):
    matrix_list = []
    for _ in range(10):
        index_array = np.arange(matrix.size)
        np.random.shuffle(index_array)
        index_array = index_array[:rate]

        mutated_matrix = np.copy(matrix)
        for m in range(rate):
            if mutated_matrix[index_array[m]] == 1:
                mutated_matrix[index_array[m]] = one_opposite
            else:
                mutated_matrix[index_array[m]] = 1
        matrix_list.append(mutated_matrix)
    return matrix_list


# read data:
data_file = open("assbio3.txt", 'r')
data = []
temp = []
for line in data_file:
    if line.rstrip() == "":
        temp = np.array(temp)
        temp = np.where(temp == 0, one_opposite, temp)
        data.append(temp)
        temp = []
    else:
        temp.append([int(bit) for bit in line.rstrip()])
temp = np.array(temp)
temp = np.where(temp == 0, one_opposite, temp)
data.append(temp)
data_file.close()

# set 10 pics of each number as vector and stuck them as matrix:
matrices = []
matrix_data = []
for d in data:
    matrix_data.append(d.reshape(100))
    if len(matrix_data) == 10:
        matrices.append(np.array(matrix_data))
        matrix_data = []

# make T matrix for each image:
matrices_t = []
for m in matrices:
    matrices_t.append(make_T(m))


# calculate variations:
scores = []
for rate in range(5, 51, 5):
    mutated_data = modificate_matrix(data[0].reshape(100), rate)
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
    print("score:" + str(scores))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

colors = ["purple", "blue", "turquoise", "lime", "greenyellow", "yellow", "khaki", "orange", "orangered", "red"]
markers = ['o', 'v', '<', '>', '^', 's', 'p', '*', 'h', 'X']

for i in range(len(colors)):
    ax.scatter(0, 0, 0, color=colors[i], marker='.', label=i+1)

for i in range(len(markers)):
    ax.scatter(0, 0, 0, color='k', marker=markers[i], label=(i + 1)*5)

for ratio in range(len(scores)):
    for number in range(len(scores[ratio])):
        ax.scatter(number, (ratio + 1) * 5, scores[ratio][number], color=colors[number], marker=markers[ratio])


ax.set_xlabel('Number Learned')
ax.set_ylabel('Mutation Ratio')
ax.set_zlabel('Euclidean Distance')
plt.legend(bbox_to_anchor=(1, 1), loc='best')
#plt.show()

