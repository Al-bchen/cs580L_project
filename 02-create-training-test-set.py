import csv
import random
from collections import Counter

random.seed('CS580L Project - Recommender Systems')  # Random seed, set to some value to keep it always same
with open('ml-modified/ratings.csv', 'r', newline='', encoding='utf8') as file_csv_input:
    csv_reader = csv.reader(file_csv_input)
    list_data = list(csv_reader)
    random.shuffle(list_data)
    list_test_set = list_data[0:5000]
    list_training_set = list_data[5000:]
print(list_training_set[0], list_test_set[0])

with open('ml-modified/training_set.csv', 'w', newline='', encoding='utf8') as file_csv_output:
    csv_writer = csv.writer(file_csv_output)
    csv_writer.writerows(list_training_set)

with open('ml-modified/test_set.csv', 'w', newline='', encoding='utf8') as file_csv_output:
    csv_writer = csv.writer(file_csv_output)
    csv_writer.writerows(list_test_set)

count = Counter([each[0] for each in list_data])
print(len(list(count)))
print(count.most_common(10))

count = Counter([each[0] for each in list_test_set])
print(len(list(count)))
print(count.most_common(10))
