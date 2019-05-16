import pandas as pd
import matplotlib.pyplot as plt
import csv
import keras
import time

n_features = 17

pd_data = pd.read_csv('ml-modified/training_set.csv', header=None)
pd_test = pd.read_csv('ml-modified/test_set.csv', header=None)
with open('ml-modified/list_id.csv', 'r', newline='', encoding='utf8') as file_csv_input:
    csv_reader = csv.reader(file_csv_input)
    list_movie_id = list(csv_reader)
    list_movie_id = [int(each[0]) for each in list_movie_id]

pd_data[1] = [list_movie_id.index(each) for each in list(pd_data[1])]
pd_test[1] = [list_movie_id.index(each) for each in list(pd_test[1])]

n_users = pd_data[0].nunique()
n_movies = pd_data[1].nunique()

time_start = time.time()

movie_in = keras.layers.Input(shape=(1,), name='movie-in')
movie_em = keras.layers.Embedding(n_movies+1, n_features, name='movie-em')(movie_in)
movie_vec = keras.layers.Flatten(name='FlattenMovies')(movie_em)

user_in = keras.layers.Input(shape=(1,), name='user-in')
user_em = keras.layers.Embedding(n_users+1, n_features, name='user-em')(user_in)
user_vec = keras.layers.Flatten(name='FlattenUsers')(user_em)

prod = keras.layers.dot([movie_vec, user_vec], axes=1, normalize=False)
model = keras.Model([user_in, movie_in], prod)
model.compile(keras.optimizers.Adam(0.001), loss='mse')

history = model.fit([pd_data[0], pd_data[1]], pd_data[2], batch_size=64, epochs=100)

predict = model.predict([pd_test[0], pd_test[1]])
real_rating = list(pd_test[2])
error = [float(abs(predict[i] - real_rating[i])) for i in range(len(predict))]

time_end = time.time()
print('Finish, used %.3lfs' % (time_end - time_start))

list_bound = [0.5 * i - 0.25 for i in range(12)]
list_bound[0] = 0
list_bound[-1] = 5
plt.figure(figsize=(6, 3.5))
print(error)
(n, bins, patches) = plt.hist(error, bins=list_bound, density=True, facecolor="blue", edgecolor="black", alpha=0.5)
for v, i in zip(n, bins):
    print(v, i)
    if i != 0:
        plt.text(i+0.25, v, str('%.4f' % v), va='bottom', ha='center')
    else:
        plt.text(i+0.125, v, str('%.4f' % v), va='bottom', ha='center')
# plt.hist(real_rating, bins=list_bound, density=True, edgecolor="black", alpha=0.5)
plt.xticks(list_bound[1:-1])
plt.title(f'Figure 5.3.2: Matrix Factorization using keras')
plt.savefig(f'figure/mf-keras-{n_features}.png')

error = [each**2 for each in error]
print(sum(error) / len(error))

plt.show()
