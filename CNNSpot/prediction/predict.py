import tensorflow as tf
from numpy.matlib import repmat

import tifffile as Image
import numpy as np
import time
start = time.time()

image_name = '/home/home/Desktop/20180731_JRT3_03_ligand.tif'
pretrained_model_dir = '/home/home/Desktop/tf_model/'
output = '/home/home/Desktop/output.tif'
kernel_size = 7
slice = 80
probability_threshold = 0
iou_threshold =0.1# 0.58
def predict():
    with tf.Graph().as_default() as graph:
      with tf.Session() as sess:
        tf.saved_model.loader.load(export_dir=pretrained_model_dir, sess=sess, tags=["serve"])

        image = Image.imread(image_name)
        #for slice in range(image.shape[0]):
        n_row, n_column = np.shape(image[slice])

        predict_windows =[]
        positions =[]
        for row in range(n_row-kernel_size+1):
          for column in range(n_column-kernel_size+1):
            predict_windows.append(np.array([image[slice, row:kernel_size+row, column:kernel_size+column]]).transpose())
            positions.append([slice, row, column, kernel_size+row, kernel_size+column])
        predict_windows =np.array(predict_windows)
        boxes = np.take(positions, [1, 2, 3, 4], axis=1)
        l_input = graph.get_tensor_by_name('myInput:0')
        l_output = graph.get_tensor_by_name('fully_connected/BiasAdd:0')
        logits = sess.run(l_output, feed_dict={l_input: predict_windows})
        probabilities = sess.run(tf.nn.softmax(logits=logits, axis=0))
        probabilities[:, 0] = np.subtract(probabilities[:, 0], probability_threshold)
        iou = sess.run(tf.image.non_max_suppression(boxes=boxes, scores= probabilities[:,0], iou_threshold=iou_threshold, max_output_size=probabilities.shape[0]))
        print(iou)
        probabilities = np.take(probabilities, iou, axis=0)
        print(probabilities)
        positions = np.take(positions, iou, axis=0)
        print(positions)
        onehot = sess.run(tf.argmax(probabilities, axis=1))
        print(onehot)
        predicted_objects =np.where(onehot == 0)
        print(predicted_objects)

        for i in range(len(predicted_objects[0])):
          save =(positions[predicted_objects[0][i]])
          image[save[0], save[1]+3:save[3]-3, save[2]+3:save[4]-3] =99999
        print('Slice done:', slice)
        sess.close()
        Image.imwrite(output, image)
    end = time.time()
    print('Performance:', np.round(end - start,2), 'seconds')


