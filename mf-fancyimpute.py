import fancyimpute
import csv
import numpy as np
import time
import matplotlib.pyplot as plt

number_of_user = 610
number_of_movie= 2269
K_of_knn = 20

time_start = time.time()


with open('ml-modified/training_set.csv', 'r', newline='', encoding='utf8') as file_csv_input:
    csv_reader = csv.reader(file_csv_input)
    list_data = list(csv_reader)

with open('ml-modified/test_set.csv', 'r', newline='', encoding='utf8') as file_csv_input:
    csv_reader = csv.reader(file_csv_input)
    list_test = list(csv_reader)

with open('ml-modified/list_id.csv', 'r', newline='', encoding='utf8') as file_csv_input:
    csv_reader = csv.reader(file_csv_input)
    list_movie_id = list(csv_reader)
    list_movie_id = [int(each[0]) for each in list_movie_id]

np_data = np.zeros((number_of_user, number_of_movie))
np_data.fill(np.nan)
for each in list_data:
    user_id = int(each[0]) - 1
    movie_id = list_movie_id.index(int(each[1]))
    rating = float(each[2])
    np_data[user_id, movie_id] = rating

print(np_data)

number_of_feature = 4

model = fancyimpute.MatrixFactorization(
    learning_rate=0.001,
    rank=number_of_feature,
    l2_penalty=0,
    min_improvement=0,
    patience=100,
    min_value=0.5,
    max_value=5,
)
np_prediction = model.fit_transform(np_data)

error = []
for each in list_test:
    predict_user_id = int(each[0]) - 1
    predict_movie_id = list_movie_id.index(int(each[1]))
    real_rating = float(each[2])

    predict_rating = np_prediction[predict_user_id][predict_movie_id]

    error.append(abs(real_rating - predict_rating))

time_end = time.time()
print('Finish, used %.3lfs' % (time_end - time_start))

list_bound = [0.5 * i - 0.25 for i in range(12)]
list_bound[0] = 0
list_bound[-1] = 5
plt.figure(figsize=(6, 3.5))
(n, bins, patches) = plt.hist(error, bins=list_bound, density=True, facecolor="blue", edgecolor="black", alpha=0.5)
for v, i in zip(n, bins):
    print(v, i)
    if i != 0:
        plt.text(i+0.25, v, str('%.4f' % v), va='bottom', ha='center')
    else:
        plt.text(i+0.125, v, str('%.4f' % v), va='bottom', ha='center')
# plt.hist(real_rating, bins=list_bound, density=True, edgecolor="black", alpha=0.5)
plt.xticks(list_bound[1:-1])

error = [each**2 for each in error]
print(sum(error) / len(error))

plt.title('Figure 5.3.1: Matrix Factorization using fancyimpute\nMSE = %.4f'%(sum(error) / len(error)))
plt.savefig(f'figure/mf-fancyimpute-{number_of_feature}-mse.png')

plt.title('Figure 5.3.1: Matrix Factorization using fancyimpute')
plt.savefig(f'figure/mf-fancyimpute-{number_of_feature}.png')

plt.show()
