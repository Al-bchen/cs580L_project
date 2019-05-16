import matplotlib.pyplot as plt
import csv
import time
import random
import copy

class random_guessing(object):
    @staticmethod
    def list_difference_real():
        with open('ml-modified/ratings.csv', 'r', newline='', encoding='utf8') as file_csv_input:
            csv_reader = csv.reader(file_csv_input)
            _list_real = list(csv_reader)
        _list_real_rating = [float(each[2]) for each in _list_real]
        _list_new_rating = copy.deepcopy(_list_real_rating)
        random.shuffle(_list_new_rating)
        return [abs(_[0] - _[1]) for _ in zip(_list_new_rating, _list_real_rating)]


list_bound = [0.5 * i - 0.25 for i in range(12)]
list_bound[0] = 0
list_bound[-1] = 5
list_count = [0 for _ in range(len(list_bound) - 1)]

list_ratings = [0.5 * _ for _ in range(1, 11)]

time_start = time.time()

list_difference_real = random_guessing.list_difference_real()

time_end = time.time()
print('Finish, used %.3lfs' % (time_end - time_start))

plt.figure(figsize=(6, 3))
(n, bins, patches) = plt.hist(list_difference_real, bins=list_bound, density=True, facecolor="blue", edgecolor="black", alpha=0.5)
for v, i in zip(n, bins):
    print(v, i)
    if i != 0:
        plt.text(i+0.25, v, str('%.4f' % v), va='bottom', ha='center')
    else:
        plt.text(i + 0.125, v, str('%.4f' % v), va='bottom', ha='center')
# plt.hist(real_rating, bins=list_bound, density=True, edgecolor="black", alpha=0.5)
plt.xticks(list_bound[1:-1])
plt.title('Figure 5.1: Random guessing')
plt.savefig('figure/random.png')

list_difference_real = [each**2 for each in list_difference_real]
print(sum(list_difference_real) / len(list_difference_real))

plt.show()
