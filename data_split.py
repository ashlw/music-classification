import math
import os
import random

composers = os.listdir("data")
old_paths = []
for c in composers:
    path = c + '/'
    paths = os.listdir('data/' + path)
    for file in paths:
        old_path = path + file
        old_paths.append(old_path)

data_size = len(old_paths)
print("You have " + str(data_size) + " samples.")
train_size = math.ceil(data_size * 0.7)
dev_size = math.ceil(data_size * 0.2)
random.shuffle(old_paths)
train_data = old_paths[0:train_size]
dev_data = old_paths[train_size+1:train_size+dev_size]
test_data = old_paths[train_size+dev_size+1:]
# os.mkdir('train')
# os.mkdir('dev')
# os.mkdir('test')


data_split = open('data_split.txt', 'w+')
data_split.write(str(train_data) + '\n')
data_split.write(str(dev_data) + '\n')
data_split.write(str(test_data) + '\n')
data_split.close()
