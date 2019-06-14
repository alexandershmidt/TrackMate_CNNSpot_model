import os
import tifffile as Image
import numpy as np

def load_labels(path_to_labels):
    false_image_list = []
    false_label = []
    true_image_list = []
    true_label = []
    false_labels = os.listdir(path_to_labels + '/labels/noise/')
    for image_index in range(len(false_labels)):
        image = Image.imread(path_to_labels + '/labels/noise/'+false_labels[image_index])
        false_image_list.append(np.array([image]).transpose())
        false_label.append([0, 1])
    false_train_x = np.array(false_image_list)
    false_train_y = np.array(false_label)

    true_labels = os.listdir(path_to_labels + '/labels/signal/')
    for image_index in range(len(true_labels)):
        image = Image.imread(path_to_labels + '/labels/signal/' + true_labels[image_index])
        true_image_list.append(np.array([image]).transpose())
        true_label.append([1, 0])
    true_train_x = np.array(true_image_list)
    true_train_y = np.array(true_label)

    train_x = np.append(false_train_x, true_train_x, axis=0)
    train_y = np.append(false_train_y, true_train_y, axis=0)
    print(len(train_y), 'labels loaded')
    return train_x, train_y