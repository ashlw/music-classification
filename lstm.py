import pretty_midi
import numpy as np
import pandas as pd
import tensorflow as tf

def readData():
    data = open('data_split.txt', 'r')
    train_data = eval(data.readline())
    dev_data = eval(data.readline())
    test_data = eval(data.readline())
    return train_data, dev_data, test_data

# returns list of midi files and their corresponding composer labels
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

def main():
    train_data, dev_data, test_data = readData()
    x_train, y_train = parseFiles(train_data)
    x_dev, y_dev = parseFiles(dev_data)
    # x_test, y_test = parseFiles(test_data)

    # parse midi into notes ( # of files, notes )
    # use composers as labels (1, # of files)

if __name__ == '__main__':
    main()
