import shutil
import os
import numpy as np
import tifffile as Image
import xml.etree.ElementTree as ET


def trackmate_xml_into_training_set(protein,kernel_size, distance_false_to_true, path_to_xml, false_multiplier, path_to_labels):
    def calibrate(root):
        res = root.findall("./Settings/ImageData")
        filename = res[0].attrib['filename']
        folder = res[0].attrib['folder']
        x_cal = np.double(res[0].attrib['pixelwidth'])
        y_cal = np.double(res[0].attrib['pixelheight'])
        width = int(res[0].attrib['width'])
        height = int(res[0].attrib['height'])
        nframes = int(res[0].attrib['nframes'])
        return filename, folder, x_cal, y_cal, width, height, nframes

    def crop(image_name, image, position_x, position_y, slice, kernel_size):
        left = position_x - kernel_size
        right = position_x + kernel_size + 1
        botton = position_y - kernel_size
        top = position_y + kernel_size + 1
        return image[slice, botton:top, left:right]

    def prepare_output_folders(path_to_labels):
        try:
            shutil.rmtree(path_to_labels+'/labels')
        except:
            pass
        os.mkdir(path_to_labels + '/labels')
        os.mkdir(path_to_labels + '/labels/signal')
        os.mkdir(path_to_labels + '/labels/noise')

    #prepare_output_folders(path_to_labels)

    kernel_size = int(kernel_size / 2)

    tree = ET.parse(path_to_xml)
    root = tree.getroot()
    image_name, path_to_image, x_cal, y_cal, width, height, nframes = calibrate(root)
    image = Image.imread(path_to_image+image_name)

    print('generating true labels...')
    positions = []
    spots = {}
    count_true = 0
    for slice in range(nframes-1):
        spots_in_frame = root.findall("./Model/AllSpots/SpotsInFrame/*[@FRAME='"+str(slice)+"']")
        for spot in range(len(spots_in_frame)):
            spot_id = spots_in_frame[spot].attrib['ID']
            position_x = int(np.round(np.divide(np.double(spots_in_frame[spot].attrib['POSITION_X']), x_cal)))
            position_y = int(np.round(np.divide(np.double(spots_in_frame[spot].attrib['POSITION_Y']), y_cal)))
            frame = int(spots_in_frame[spot].attrib['FRAME'])
            Image.imwrite(path_to_labels + '/labels/signal/label_'+str(slice)+'_' + spot_id + protein + '.tif', crop(image_name, image, position_x, position_y, slice, kernel_size))
            positions.append([position_x, position_y])
            count_true += 1
        spots[slice] = positions
        positions = []
    print(count_true, 'signal labels generated')

    print("generating false labels")
    count_false = 0
    while count_false < false_multiplier*count_true:
        false_position_x = np.random.randint(low=kernel_size, high=width-kernel_size)
        false_position_y = np.random.randint(low=kernel_size, high=height-kernel_size)
        random_frame = np.random.randint(low=0, high=nframes-1)
        if not spots[random_frame]:
            count_false += 1
            Image.imwrite(path_to_labels + '/labels/noise/label_' + str(random_frame) + '_' + str(count_false) + protein + '.tif', crop(image_name, image, false_position_x, false_position_y, random_frame, kernel_size))
        else:
            distance_not_ok=np.sum(np.power(np.subtract(spots[random_frame], [false_position_x, false_position_y]), 2),  axis=1) < distance_false_to_true
            if not any(distance_not_ok):
                count_false += 1
                Image.imwrite(path_to_labels + '/labels/noise/label_' + str(random_frame) + '_' + str(count_false) + protein + '.tif', crop(image_name, image, false_position_x, false_position_y, random_frame, kernel_size))
    print(count_false, 'noise labels generated')
    print(count_false+count_true, 'total labels generated')