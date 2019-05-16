import csv
import itertools
import time
from collections import Counter

with open('ml-latest-small/ratings.csv', 'r', newline='', encoding='utf8') as file_csv_input:
    print('Start reading!')
    time_start = time.time()
    csv_reader = csv.reader(file_csv_input)
    list_data = list(csv_reader)
    list_data = list_data[1:]
    print(f'Total ratings count: {len(list_data)}')

    count_movie_rated = Counter([each[1] for each in list_data])
    count_movie_10 = 0
    list_id_more_than_10_rating = []
    for each in count_movie_rated.items():
        if each[1] >= 10:
            list_id_more_than_10_rating.append(each[0])
            count_movie_10 += 1
    print(f'Movies more than 10 ratings: {count_movie_10}')

    list_data = [each for each in list_data if each[1] in list_id_more_than_10_rating]
    print('Movie that rated 10 times less removed')
    print(f'Total ratings count: {len(list_data)}')

    # count_user_rated = Counter([each[0] for each in list_data])
    # count_user_10 = 0
    # list_id_more_than_10_rating = []
    # for each in count_user_rated.items():
    #     if each[1] >= 0:
    #         list_id_more_than_10_rating.append(each[0])
    #         count_user_10 += 1
    # print(f'Users more than 10 ratings: {count_user_10}')
    #
    # list_data = [each for each in list_data if each[0] in list_id_more_than_10_rating]
    # print('User that rated 10 times less removed')
    # print(f'Total ratings count: {len(list_data)}')

    time_end = time.time()
    print('End reading! Used time: %.3lfs' % (time_end - time_start))

list_data = [each[:-1] for each in list_data]

with open('ml-modified/ratings.csv', 'w', newline='', encoding='utf8') as file_csv_output:
    csv_writer = csv.writer(file_csv_output)
    csv_writer.writerows(list_data)

count_movie_rated = Counter([each[1] for each in list_data])
list_movie_id = list(count_movie_rated)
list_movie_id = [int(each) for each in list_movie_id]
list_movie_id.sort()
print(list_movie_id)
with open('ml-modified/list_id.csv', 'w', newline='', encoding='utf8') as file_csv_output:
    csv_writer = csv.writer(file_csv_output)
    for each in list_movie_id:
        csv_writer.writerow([each])

print('finish!')
