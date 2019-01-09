from music21 import converter, instrument, note, chord
from google.cloud import storage
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Activation
import os
import glob
import numpy
from keras.utils import np_utils
#----Global Variables----
current_epoch = 1
GCSPath = "gs://music-lstm"
input_length = 100
n_vocab = 0
#----Function to fetch midi files from Google Cloud Storage----
def download_files_from_gcs():
    print("Initializing client")
    storage_client = storage.Client("is-music-lstm")
    print("Creating Bucket")
    bucket = storage_client.get_bucket("music-lstm")
    print("Creating Directory")
    dirName = 'midi_songs'
    if not os.path.exists(dirName):
        # Create target Directory
        os.mkdir(dirName)
        print("Directory " +  dirName +  " Created ") 
    else:
        print("Directory " + dirName +  " already exists")
    # Save file 
    print("Loading midi songs from gcs")
    blobs=list(bucket.list_blobs(prefix="midi_songs"))
    for blob in blobs:
        if(not blob.name.endswith("/")):
            blob.download_to_filename(blob.name)
    print("Finsihed loading songs from gcs")
#----Function to read and parse notes from midi files
def prepare_data():
    global n_vocab
    logger = open("testfile.txt", "w")

    notes = []
    print("Preparing data")
    for file in glob.glob("midi_songs/*.mid"):
        midi = converter.parse(file)
        print("Reading %s" % file)
        notes_to_parse = None
        #To ensure notes and chords are from 1 instrument only
        try:
            notes_to_parse = instrument.partitionByInstrument(midi).parts[0].recurse() 
        except: 
            notes_to_parse = midi.flat.notes

        for element in notes_to_parse:
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                notes.append('.'.join(str(n) for n in element.normalOrder))
    print(notes)
    logger.write("This is notes")
    logger.write(str(notes))
    n_vocab = len(set(notes))
    pitchnames = sorted(set(item for item in notes))
     # create a dictionary to map pitches to integers
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))
    network_input = []
    network_output = []
    # create input sequences and the corresponding outputs
    for i in range(0, len(notes) - input_length, 1):
        sequence_in = notes[i:i + input_length]
        sequence_out = notes[i + input_length]
        network_input.append([note_to_int[char] for char in sequence_in])
        network_output.append(note_to_int[sequence_out])
    n_patterns = len(network_input)
    # reshape the input into a format compatible with LSTM layers
    network_input = numpy.reshape(network_input, (n_patterns, input_length, 1))
    # normalize input
    network_input = network_input / float(n_vocab)
    network_output = np_utils.to_categorical(network_output)
    print("VOCAB")
    print(n_vocab)
    return (network_input, network_output)
def create_model(network_input):
    model = Sequential()
    model.add(LSTM(
        512,
        input_shape=(network_input.shape[1], network_input.shape[2]),
        return_sequences=True
    ))
    model.add(Dropout(0.3))
    model.add(LSTM(512, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(512))
    model.add(Dense(256))
    model.add(Dropout(0.3))
    model.add(Dense(n_vocab))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    return model
if __name__ == "__main__":
    download_files_from_gcs()
    input_data, labels = prepare_data()
    model = create_model(input_data)
