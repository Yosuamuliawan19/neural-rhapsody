# for reading in music
from music21 import instrument, note, stream, chord
# to load the keras models
from keras import backend as K
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, LSTM, Activation
# for flask server
import json
import requests
from functools import update_wrapper
from flask import Flask, make_response, request, current_app  
# etc
from datetime import timedelta  
import pickle
import numpy


app = Flask(__name__)

#to enable crossdomain header
def crossdomain(origin=None, methods=None, headers=None, max_age=21600, attach_to_all=True, automatic_options=True):  
    if methods is not None:
        methods = ', '.join(sorted(x.upper() for x in methods))
    if headers is not None and not isinstance(headers, basestring):
        headers = ', '.join(x.upper() for x in headers)
    if not isinstance(origin, basestring):
        origin = ', '.join(origin)
    if isinstance(max_age, timedelta):
        max_age = max_age.total_seconds()
    def get_methods():
        if methods is not None:
            return methods
        options_resp = current_app.make_default_options_response()
        return options_resp.headers['allow']
    def decorator(f):
        def wrapped_function(*args, **kwargs):
            if automatic_options and request.method == 'OPTIONS':
                resp = current_app.make_default_options_response()
            else:
                resp = make_response(f(*args, **kwargs))
            if not attach_to_all and request.method != 'OPTIONS':
                return resp
            h = resp.headers
            h['Access-Control-Allow-Origin'] = origin
            h['Access-Control-Allow-Methods'] = get_methods()
            h['Access-Control-Max-Age'] = str(max_age)
            if headers is not None:
                h['Access-Control-Allow-Headers'] = headers
            return resp
        f.provide_automatic_options = False
        return update_wrapper(wrapped_function, f)
    return decorator



#function to prepare seeds from available songs
#returns a section of 100 notes from a random song
def makeSeeds(notes):
    # creeate mapper to map notes to int
    notesAmount = len(set(notes))
    mapper = dict((note, number) for number, note in enumerate(sorted(set(item for item in notes))))
    # create a lot of [100] array from the notes
    input_sequence_length = 100
    input_data = []
    for idx in range(0,len(notes)- input_sequence_length,1):
        current_input = notes[idx:idx+ input_sequence_length]
        input_data.append( [mapper[char] for char in current_input] ) 
    # choose a random index for the seed of the song
    randomIndex = numpy.random.randint(0, len(input_data)-1)
    choosenSeed = input_data[randomIndex]
    return choosenSeed

def generate_notes(model, notes):
    # creeate mapper to map notes to int
    notesAmount = len(set(notes))
    vocab_notes = sorted(set(item for item in notes))
    mapper = dict((number, note) for number, note in enumerate(vocab_notes))
    pattern = makeSeeds(notes)
    prediction_output = []
    print("Generating notes, please wait")
    # generate 300 notes
    for note_index in range(300):
        print("Generating note " + str(note_index) + " /")
        # reshaping previous input into input suitable for LSTM network
        prediction_input  =numpy.reshape(pattern, (1, len(pattern), 1))
        prediction_input  =prediction_input/float(notesAmount)
        # getting next note
        prediction = model.predict(prediction_input, verbose=0)
        index = numpy.argmax(prediction)
        # removing the first one
        pattern = pattern[1: len(pattern)]
        # mapping index to note
        result = mapper[index]
        # append to prediction output
        prediction_output.append(result)
        # adding previous prediction to next input
        pattern.append(index)
        print(prediction_output)
    print("Finish generating note")
    print("Saving to midi file")
    # create variables to store midi and current tinme
    song_prediction = []
    currentTime = 0
    for nextPart in prediction_output:
        # if nextPart is a chord
        if ('.'  in nextPart) or nextPart.isdigit():
            # seperate notes in chords and put in content
            content = []
            notes_in_chord = nextPart.split('.')
            for current_note in notes_in_chord:
                next_note = note.Note( int(current_note))
                next_note.storedInstrument = instrument.Piano()
                content.append(next_note)
            # create a chord object from content
            next = chord.Chord(content)
            # add offset to prevent all chords being at the same time
            next.offset = currentTime
            # append to song_prediction
            song_prediction.append(next)
            currentTime = currentTime +  0.5
        else:
            # if its note, just wrap it in a note object
            next_note = note.Note(nextPart)
            # add offset
            next_note.offset = currentTime
            currentTime = currentTime +  0.5
            next_note.storedInstrument = instrument.Piano()
            # append to song_prediction
            song_prediction.append(next_note)
    mStream = stream.Stream(song_prediction)
    mStream.write('midi',fp='generated.mid')
# function to serve API in flask
# /generate/Beethoven -> to generate Beethoven songs
# /generate/Mozart -> to generate Mozart songs
# /generate/Chopin -> to generate Chopin songs
@app.route('/generate/<artist>')
@crossdomain(origin='*')
def generate(artist):
    artist = str(artist)
    print("Generating song based on " + artist)
    #load the notes used to train the model
    if artist == 'Beethoven':
        print("Using Beethoven model and notes")
        modelPath = "models/model_beethoven.h5"
        notesPath = 'data/notes_beethoven'
    elif artist == 'Chopin':
        print("Using Chopin model and notes")
        modelPath = "models/model_chopin.h5"
        notesPath = 'data/notes_chopin'
    elif artist == 'Mozart':
        print("Using Mozart model and notes")
        modelPath = "models/model_mozart.h5"
        notesPath = 'data/notes_mozart'
    else :
        return "artist not valid"
    # loading the respective model and noteslist for each artist
    print("Loading model for " + artist)
    model = load_model(modelPath)
    print("Loading notes list for " + artist)
    notesList = []
    with open(notesPath, 'rb')as path_to_note:
        notesList = pickle.load(path_to_note)   
    # generate song 
    generate_notes(model, notesList)
    # clearing keras session for the next song
    print("Clearing keras session")
    K.clear_session()
    # post to file.io to upload file
    print("Posting to file.io")
    r = requests.post("https://file.io/?expires=1w", files = {'file': open('generated.mid', "rb")})
    # return the response of file.io
    print("Returning response")
    print(r.text)
    return r.text
# function to server flask server
if __name__ == '__main__':
    app.run(host= '0.0.0.0',port=5000, debug=True)