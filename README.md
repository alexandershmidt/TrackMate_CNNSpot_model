# TrackMate_CNNSpot_model
CNN model

Prerequisites for Python scripts: Python 3.7, Anaconda 2019.03, shutil, numpy, os, time, tiffile, 
xml.etree.ElementTree, sklearn.model_selection, tensorflow, math, tensorflow.python.saved_model.simple_save

Train model:
User defined parameters: 
1) path to labels - training set
2) kernel_size - size of each label
3) distance_false_to_true - minimum distance to all signals - important for generating noises
4) select false_multiplier - proportion signal: noise labels
5) select test_size : proportion training set: test set
6) select minibatch_size 
7) select learning_rate
8) select number of iterations - epochs
9) select output folder - (.pb file)
10) run train

Example is provided in run_training.py
