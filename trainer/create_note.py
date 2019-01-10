import glob
import pickle
import numpy
import h5py
import argparse
import os
from datetime import datetime
import tensorflow as tf
from tensorflow.python.client import device_lib
from tensorflow.python.lib.io import file_io
from google.cloud import storage
from music21 import converter, instrument, note, chord
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Dropout, Activation
from keras.utils import np_utils, multi_gpu_model
from keras.callbacks import ModelCheckpoint, TensorBoard, LambdaCallback, Callback
from keras import backend as K

#----Global Variables----
current_epoch = 0
GCSPath = "gs://music-lstm"
input_length = 100
n_vocab = 0
songDir = 'midiChopin'
songType = 'Chopin'
#----Function to fetch midi files from Google Cloud Storage----
def download_files_from_gcs():
    print("Initializing client")
    storage_client = storage.Client("is-music-lstm")
    print("Creating Bucket")
    bucket = storage_client.get_bucket("music-lstm")
    print("Creating Directory")
    dirName = songDir
    if not os.path.exists(dirName):
        # Create target Directory
        os.mkdir(dirName)
        print("Directory " +  dirName +  " Created ") 
    else:
        print("Directory " + dirName +  " already exists")
    # Save file 
    print("Loading midi songs from gcs")
    blobs=list(bucket.list_blobs(prefix="midi-chopin"))
    count = 1
    for blob in blobs:
        if(not blob.name.endswith("/")):
            blob.download_to_filename(songDir + "/" + str(count)+  " .mid")
            count = count + 1
    print("Finsihed loading songs from gcs")
#----Function to read and parse notes from midi files----
def prepare_data():
    global notesAmount
    logger = open("testfile.txt", "w")
    notes = []
    print("Preparing data")
    for file in glob.glob(songDir + "/*.mid"):
        midi = converter.parse(file)
        print("Reading %s" % file)
        song = None
        #To ensure notes and chords are from 1 instrument only
        try:
            song = instrument.partitionByInstrument(midi).parts[0].recurse() 
        except: 

            song = midi.flat.notes
        for part in song:
            if isinstance(part, note.Note):
                # if its just a note, append the part's pitch
                notes.append(str(part.pitch))
            elif isinstance(part, chord.Chord):
                # it its a chord, append in an encoded form
                # e.g chord that contains B5 and C3, becomes 1.9, because B5 is mapped to 1, and C3 is mapped to 9
                notes.append('.'.join(str(n)for  n in part.normalOrder))
    # saving notes
    with open('data/notes' + songType, 'wb') as filepath:
        pickle.dump(notes, filepath)
    # printing to termina;
    print("This is the list of notes in the training data")
    print(notes)
    logger.write(str(notes))
    # create mapper
    notesAmount = len(set(notes))
    pitchnames = sorted(set(item for item in notes))
    mapper = dict((note, number) for number, note in enumerate(pitchnames))
    # create variables to store network input and output
    input_data = []
    label = []
    # create input sequences and the corresponding outputs
    print("Encoding the input into  [100] array and output to a label")
    for i in range(0, len(notes) - input_length, 1):
        sequence_out = notes[i + input_length]
        sequence_in = notes[i:i + input_length]
        label.append(mapper[sequence_out])
        input_data.append([mapper[char] for char in sequence_in])
    # one hot encode to label
    label = np_utils.to_categorical(label)
    # reshape the input into a format compatible with LSTM layers
    input_data = numpy.reshape(input_data, ( len(input_data), input_length, 1))
    # normalize input
    input_data = input_data / float(notesAmount)
    print("Amount of unique notes in input" + str(notesAmount))
    return (input_data, label)

if __name__ == '__main__':
    download_files_from_gcs()
    input_data, labels = prepare_data()
