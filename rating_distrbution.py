import matplotlib.pyplot as plt
import csv

with open('ml-modified/training_set.csv', 'r', newline='', encoding='utf8') as file_csv_input:
    csv_reader = csv.reader(file_csv_input)
    list_data = list(csv_reader)

list_ratings = [float(each[2]) for each in list_data]
print(list_ratings)

bins = [0.5 * i + 0.25 for i in range(11)]

plt.figure(figsize=(6, 3.5))
(n, bins, patches) = plt.hist(list_ratings, bins=bins, facecolor="blue", edgecolor="black", alpha=0.5)

for v, i in zip(n, bins):
    print(v, i)
    plt.text(i+0.25, v, str('%d' % v), va='bottom', ha='center')
plt.xticks([each + 0.25 for each in bins[:-1]])

print(len([each for each in list_ratings if each == 5]))
plt.title('Figure 4: Rating distribution')
plt.savefig('figure/rating_distribution.png')
plt.show()
