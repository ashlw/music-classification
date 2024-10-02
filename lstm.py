import pretty_midi
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import LSTM, Dense, Dropout
import os
# https://github.com/ruohoruotsi/LSTM-Music-Genre-Classification/blob/master/lstm_genre_classifier_pytorch.py

def readData():
    data = open('data_split.txt', 'r')
    train_data = eval(data.readline())
    dev_data = eval(data.readline())
    test_data = eval(data.readline())
    return train_data, dev_data, test_data

# returns list of midi files(x) and their corresponding composer labels(y)
def parseFiles(data):
    x, y = [], []
    for d in data:
        path = os.path.split(d)
        # HARDCODING IS BAD I KNOW SORY
        if path[0] == 'bach':
            y.append(0)
        elif path[0] == 'beethoven':
            y.append(1)
        elif path[0] == 'chopin':
            y.append(2)
        else:
            y.append(3)
        path = "data/" + d
        x.append(pretty_midi.PrettyMIDI(path))

    return x, np.array(y)

# https://www.tensorflow.org/tutorials/audio/music_generation#create_the_training_dataset
def midi_to_notes(pm: pretty_midi):
  # pm = pretty_midi.PrettyMIDI(midi_file)
  notes = []
  for instrument in pm.instruments:
      for note in instrument.notes:
          # Each note has a start time, end time, and pitch
          notes.append([note.start, note.end, note.pitch, note.velocity])
  # Sort notes by start time to ensure chronological order
  notes.sort(key=lambda x: x[0])
  return notes

def create_seq(notes: list, composer: int, seq_len: int):
    # modify this part for more features
    input_sequences = []
    label = []
    pitches = [note[2] for note in notes]
    for i in range(0, len(pitches) - seq_len):
        input_sequences.append(pitches[i:i + seq_len])
        label.append(composer)

    return input_sequences, label

# create seq and labels
def label(data: list, y: list, seq_length: int):
    # for each midi and corresponding composer label
    total_seq = []
    labels = []
    for i in range(len(data)):
        # extract features
        pm = midi_to_notes(data[i])
        seq, label = create_seq(pm, y[i], seq_length)
        total_seq.extend(seq)
        labels.extend(label)
        # num_sequences, sequence_length
    total_seq = np.array(total_seq)
    total_seq = np.reshape(total_seq, (total_seq.shape[0], total_seq.shape[1], 1))
    labels = np.array(labels)
    return total_seq, labels


def main():
    train_data, dev_data, test_data = readData()
    x_train, y_train = parseFiles(train_data)
    x_dev, y_dev = parseFiles(dev_data)
    # x_test, y_test = parseFiles(test_data)

    # HYPERPARAMS
    seq_length = 50
    batchsize = 70  # num of training examples per minibatch
    num_epochs = 100
    lr = 0.001

    # split into sequences and corresponding composers
    x_train, y_train = label(x_train, y_train, seq_length)
    x_dev, y_dev = label(x_dev, y_dev, seq_length)

    model = Sequential()
    model.add(LSTM(128, input_shape=(seq_length, 1), return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(128))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(4, activation='softmax'))  # num_composers = 4

    opt = Adam(learning_rate=lr)
    # categorical_crossentropy requires encoded labels, i can change this in the future
    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=num_epochs, batch_size=batchsize, validation_data=(x_dev, y_dev))

    # Evaluate
    test_loss, test_accuracy = model.evaluate(x_dev, y_dev)
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")


if __name__ == '__main__':
    main()
