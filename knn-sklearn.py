import os
import csv
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

number_of_user = 610
number_of_movie= 2269
K_of_knn = 20


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
for each in list_data:
    user_id = int(each[0]) - 1
    movie_id = list_movie_id.index(int(each[1]))
    rating = float(each[2])
    np_data[user_id, movie_id] = rating

print(np_data)

time_start = time.time()
model = NearestNeighbors(n_neighbors=K_of_knn+1, metric='cosine', algorithm='brute', n_jobs=-1)
model.fit(np_data)


error = []
list_test.sort(key= lambda x: x[0])
last_user_id = -1
for each in list_test:
    predict_user_id = int(each[0]) - 1
    predict_movie_id = list_movie_id.index(int(each[1]))
    real_rating = float(each[2])

    if last_user_id != predict_user_id:
        last_user_id = predict_user_id
        knn_result = model.kneighbors([np_data[predict_user_id]], return_distance=True)

    sum_of_rating = 0
    sum_of_inverse_distance = 0
    for i in range(0, K_of_knn + 1):
        current_distance = knn_result[0][0][i]
        current_rating = np_data[knn_result[1][0][i]][predict_movie_id]
        if current_rating > 0:
            sum_of_inverse_distance += 1 / current_distance
            sum_of_rating += current_rating / current_distance

    if sum_of_inverse_distance > 0:
        predict_rating = sum_of_rating / sum_of_inverse_distance
    else:
        predict_rating = 2.5

    error.append(abs(real_rating - predict_rating))
    if len(error) % 10 == 0:
        print(len(error))

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
plt.title('Figure 5.2.2: User-Based k-NN using scikit-learn')
plt.savefig('figure/knn-sklearn.png')

error = [each**2 for each in error]
print(sum(error) / len(error))

plt.show()



