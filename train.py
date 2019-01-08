from music21 import converter, instrument, note, chord

song_directory = "midi_songs/"

def read_notes():
   result = []
   for song in glob.glob(song_directory + "*.mid"):
       data = converter.parse(song)
       print("Parsing %s :" % file)
       parsing = None
       try :
           parsing = instrument.partitionByInstrument(data).parts[0].recurse()
       except:
           parsing = data.flat.notes
       for sample in parsing
           if isinstance(sample, note.Note):
               result.append(str(sample.pitch))
           elif isinstance(sample, chord.Chord):
               result.append('.'.join(str(n) for n in element.normalOrder))
   return result


if __name__ == "__main__":

    notes = read_notes()
