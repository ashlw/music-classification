import pretty_midi
import numpy as np
import pandas as pd
import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F

#ref: # https://github.com/ruohoruotsi/LSTM-Music-Genre-Classification/blob/master/lstm_genre_classifier_pytorch.py

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
        y.append(path[0])
        path = "data/" + d
        x.append(pretty_midi.PrettyMIDI(path))

    return x, np.array(y)

# from https://www.tensorflow.org/tutorials/audio/music_generation#create_the_training_dataset
def midi_to_notes(midi_file: str) -> pd.DataFrame:
  pm = pretty_midi.PrettyMIDI(midi_file)
  instrument = pm.instruments[0]
  notes = collections.defaultdict(list)

  # Sort the notes by start time
  sorted_notes = sorted(instrument.notes, key=lambda note: note.start)
  prev_start = sorted_notes[0].start

  for note in sorted_notes:
    start = note.start
    end = note.end
    notes['pitch'].append(note.pitch)
    notes['start'].append(start)
    notes['end'].append(end)
    notes['step'].append(start - prev_start)
    notes['duration'].append(end - start)
    prev_start = start

  return pd.DataFrame({name: np.array(value) for name, value in notes.items()})

class LSTM(nn.Module):
    # input dim -> number of features
    # output dim -> number of classes
    # hidden dim -> number of nodes in unit, equiv to CNN neurons
    def __init__(self, input_dim, hidden_dim, batch_size, output_dim=4, num_layers=2):
        super(LSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_layers = num_layers
        # setup LSTM layer
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers)
        # setup output layer
        self.linear = nn.Linear(self.hidden_dim, output_dim)

    def forward(self, input, hidden=None):
        lstm_out, hidden = self.lstm(input, hidden)
        logits = self.linear(lstm_out[-1])
        scores = F.log_softmax(logits, dim=1)
        return scores, hidden

    def get_accuracy(self, logits, target):
        corrects = (
                torch.max(logits, 1)[1].view(target.size()).data == target.data
        ).sum()
        accuracy = 100.0 * corrects / self.batch_size
        return accuracy.item()

def main():
    train_data, dev_data, test_data = readData()
    x_train, y_train = parseFiles(train_data)
    x_dev, y_dev = parseFiles(dev_data)
    # x_test, y_test = parseFiles(test_data)

    # TODO: extract features

    # convert to tensors
    x_train = torch.from_numpy(x_train).type(torch.Tensor)
    x_dev = torch.from_numpy(x_dev).type(torch.Tensor)
    # x_test = torch.from_numpy(x_test).type(torch.Tensor)

    # Targets is a long tensor of size (N,) which tells the true class of the sample.
    y_train = torch.from_numpy(y_train).type(torch.LongTensor)
    y_dev = torch.from_numpy(y_dev).type(torch.LongTensor)
    # y_test = torch.from_numpy(y_test).type(torch.LongTensor)

    batch_size = 35  # num of training examples per minibatch
    num_epochs = 100

    # initialize model

if __name__ == '__main__':
    main()
