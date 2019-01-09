from music21 import converter, instrument, note, chord
from google.cloud import storage
import os

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

if __name__ == "__main__":

    download_files_from_gcs()
