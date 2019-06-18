# TrackMate_CNNSpot_model
CNN model

Prerequisites for Python scripts: Python 3.7, Anaconda 2019.03, shutil, numpy, os, time, tiffile, 
xml.etree.ElementTree, sklearn.model_selection, tensorflow, math, tensorflow.python.saved_model.simple_save

Train model:
User defined parameters: 
  path to labels - training set
  kernel_size - size of each label
  distance_false_to_true - minimum distance to all signals - important for generating noises
  4select false_multiplier - proportion signal: noise labels
  test_size : proportion training set: test set
  minibatch_size 
  learning_rate
  number of iterations - epochs
  output folder - (.pb file)

run train.py

Example is provided in run_training.py
