import os
import pretty_midi
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import argparse

DEV = False

# refactor (perhaps parse files in a diff file to speed things up)
# returns an array of midi data given composer
def parseFiles(composer):
    path = "data/" + composer + '/'
    paths = os.listdir(path)
    midi_data = []
    for p in paths:
        file = path + p
        try:
            midi_data.append(pretty_midi.PrettyMIDI(file))
        except:
            print("Error with " + file)
    return midi_data

# take the mean of pitch frequencies that is weighted by note duration
def extract(midi_list):
    avg_pitches = []
    for midi in midi_list:
        hist = midi.get_pitch_class_histogram(use_duration=True)
        avg_pitches.append(np.mean(hist))
    return avg_pitches

# given a list of composers, transform into 0s and 1s for binary classification
# 1 as the desired composer
def classify(composer, arr):
    arr = np.where(arr==composer, 1, 0)
    return arr

def log_reg(x_train, x_test, y_train, y_test):
    log_reg = LogisticRegression()
    log_reg.fit(x_train, y_train)
    y_pred = log_reg.predict(x_test)
    score = accuracy_score(y_test, y_pred)
    return score

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-dev", action="store_true")
    args = parser.parse_args()
    if args.dev:
        print("IN DEV MODE")
        DEV = True

    composers = os.listdir("data")
    x_data, y = [],[]

    for c in composers:
        midi_data = parseFiles(c)
        length = len(midi_data)
        y.extend([c] * length)
        x_data.extend(extract(midi_data))
    y_data = np.array(y)
    x_data = np.array(x_data)
    x_data = x_data.reshape(-1, 1)

    # split 90/10 for train+dev/test
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.1, shuffle=True)
    # split again for the 70/20 train/dev
    x_train, x_dev, y_train, y_dev = train_test_split(x_train, y_train, test_size=0.22, shuffle=True)
    if DEV == True:
        x_test = x_dev
        y_test = y_dev

    for c in composers:
        y_train_c = classify(c, y_train)
        y_test_c = classify(c, y_test)
        score = log_reg(x_train, x_test, y_train_c, y_test_c)
        print("(log_reg) Accuracy score for %s is %f" % (c, score))


if __name__ == '__main__':
    main()
