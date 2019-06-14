import time
from preprocessing.load_labels import load_labels
from preprocessing.split_labels import split_labels
from preprocessing.trackmate_xml_into_training_set import trackmate_xml_into_training_set
from training.train import train

kernel_size = 7
distance_false_to_true = 7

path_to_labels = 'C:/Users/alex/Desktop'
false_multiplier = 1
test_size =0.4
minibatch_size = 2000
learning_rate = 0.001
iterations = 25
path_to_model ='C:/Users/alex/Desktop/tf_model'

path_to_xml = 'C:/Users/alex/Desktop/xmls/ligand_mek/20180731_JRT3_03_ligand_paper.xml'
trackmate_xml_into_training_set('ligand', kernel_size=kernel_size, distance_false_to_true=distance_false_to_true,
                                path_to_xml=path_to_xml, false_multiplier=false_multiplier,
                                path_to_labels=path_to_labels)
# path_to_xml = 'C:/Users/alex/Desktop/xmls/ligand_mek/20180731_JRT3_03_MEK_background_subtracted_paper.xml'
# trackmate_xml_into_training_set('mek', kernel_size=kernel_size, distance_false_to_true=distance_false_to_true,
#                                path_to_xml=path_to_xml, false_multiplier=false_multiplier,
#                                path_to_labels=path_to_labels)
# path_to_xml = 'C:/Users/alex/Desktop/xmls/grb2_erk/20180718-02-cell1-grb2_paper.xml'
# trackmate_xml_into_training_set('grb2', kernel_size=kernel_size, distance_false_to_true=distance_false_to_true,
#                                path_to_xml=path_to_xml, false_multiplier=false_multiplier,
#                                path_to_labels=path_to_labels)
# path_to_xml = 'C:/Users/alex/Desktop/xmls/grb2_erk/20180718-02-cell1-erk_paper.xml'
# trackmate_xml_into_training_set('erk', kernel_size=kernel_size, distance_false_to_true=distance_false_to_true,
#                                path_to_xml=path_to_xml, false_multiplier=false_multiplier,
#                                path_to_labels=path_to_labels)

train_x, train_y = load_labels(path_to_labels)

train_x_split, train_y_split, test_x_split, test_y_split = split_labels(train_x, train_y,
                                                                        test_size=test_size)
start = time.time()
train(train_x_split.transpose(), train_y_split.transpose(), test_x_split.transpose(), test_y_split.transpose(), learning_rate=learning_rate,
      iterations=iterations, minibatch_size=minibatch_size, path_to_model=path_to_model)
end = time.time()
print(start)
print(end)
print(end - start)