# Neural Rhapsody
Neural rhapsody is a neural network web app that uses an LSTM network to generate midi files using styles from customizable artist


![alt text](https://www.telegraph.co.uk/content/dam/music/2015-10/Oct29/freddie-mercury-li_1988517a_trans_NvBQzQNjv4BqqQOfaCczG4orCRBni8KzmGpkpi6EAZPmEJ8FOCKQZRw.jpg?imwidth=450)
# Features!
  - Generate any length of music you want
  - Use your own training set to generate songs according to artist style you want
  - Built in integration to Google Cloud Platform ML Engine
You can also:
  - Store your models in Google Cloud Platform Storage Bucket
  - Serve your models as web REST API


### Tech

Neural Rhapsody uses a number of open source projects to work properly:

* [Google Cloud Platform](https://cloud.google.com/) - Cloud platform to train and save  ML models 
* [Music21](http://web.mit.edu/music21/) - Python library to parse and save midi files
* [Keras](http://keras.io/) - http://keras.io/
* [Tensorflow](https://www.tensorflow.org/) - An open source machine learning framework for everyone.

### Installation

Neural Rhapsody requires pip installation to run

Install the dependencies and start the server.
Or train your model

```sh
$ cd neural-rhapsody
$ pip install keras
$ pip install music21
$ pip install tensorflow
$ pip install google-cloud-storage
```
To start api server

```sh
$ python api_server.py
```
To start training model
```sh
$ python trainer/train.py
```
Full documentation on how to train your model is available on train.py
To train using ML engine
```sh
export GOOGLE_APPLICATION_CREDENTIALS=<DIRECTORY_OF_YOUR_CLOUD_CREDENTIALS_JSON>

gcloud ml-engine jobs submit training <JOB_NAME> --module-name=trainer.train --package-path=./trainer --job-dir=<YOUR_GCS_BUCKET> --region=us-central1 --config=trainer/cloudml-gpu.yaml
```

