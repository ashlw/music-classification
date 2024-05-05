import os
import pretty_midi
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import argparse

def readData():
    data = open('data_split.txt', 'r')
    train_data = eval(data.readline())
    dev_data = eval(data.readline())
    test_data = eval(data.readline())
    return train_data, dev_data, test_data

def parseFiles(data):
    x, y = [], []
    for d in data:
        path = os.path.split(d)
        y.append(path[0])
        path = "data/" + d
        x.append(pretty_midi.PrettyMIDI(path))

    return x, np.array(y)

def extract(midi_list):
    features = []
    for midi in midi_list:
        matrix = midi.get_pitch_class_transition_matrix()
        matrix = matrix.flatten()
        features.append(matrix)
    return features

# given a list of composers, transform into 0s and 1s for binary classification
# 1 as the desired composer
def classify(composer, arr):
    arr = np.where(arr==composer, 1, 0)
    return arr

def svm(x_train, x_test, y_train, y_test):
    for kernel in ("linear", "rbf", "poly", "sigmoid"):
        svm = SVC(kernel=kernel, gamma=10)
        svm.fit(x_train, y_train)
        y_pred = svm.predict(x_test)
        score = accuracy_score(y_test, y_pred)
        print(score)
    # return score

def main():
    DEV = False
    parser = argparse.ArgumentParser()
    parser.add_argument("-dev", action="store_true")
    args = parser.parse_args()
    if args.dev:
        print("IN DEV MODE")
        DEV = True

    train_data, dev_data, test_data = readData()
    x_train, y_train = parseFiles(train_data)
    x_dev, y_dev = parseFiles(dev_data)
    x_test, y_test = parseFiles(test_data)
    print('Finished parsing')

    x_train = np.array(extract(x_train))
    x_dev = np.array(extract(x_dev))
    x_test = np.array(extract(x_test))
    composers = os.listdir("data")

    if DEV:
        print('dev')
        x_test = x_dev
        y_test = y_dev

    print("In the order of linear, rbf, poly, sigmoid")
    for c in composers:
        print("=== COMPOSER: " + c + " ===")
        y_train_c = classify(c, y_train)
        y_test_c = classify(c, y_test)
        svm(x_train, x_test, y_train_c, y_test_c)


if __name__ == '__main__':
    main()
