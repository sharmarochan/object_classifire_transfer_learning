
# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")
from vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical
# from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import cv2
import os
from keras.layers import merge, Input
from keras.layers import Dense, Activation, Flatten
from keras.models import Model
from keras.utils import np_utils
from keras.applications.imagenet_utils import preprocess_input
import time
from sklearn.metrics import confusion_matrix
from keras.models import load_model
import imutils
import time
import datetime
from shutil import copyfile

global now
now = datetime.datetime.now()
now = now.strftime("%Y%m%d%H%M")
global classify_dir_savepath


def classify_main(selected_model_path, trained_text_file, folder_path_classify):
    classify_dir_savepath = r"D:\Dev_Con_Project\wri_devcon_object_detection"
    classify_dir_savepath = os.path.join(classify_dir_savepath, str(now))
    os.mkdir(classify_dir_savepath)                                         #create dir according to time




    print("[INFO] loading network...")
    model = load_model(selected_model_path)
    all_image_paths = []

    for dir_, _, files in os.walk(folder_path_classify):     # Gets all the image paths for the training of a model
        for img_name in files:
            relDir = os.path.relpath(dir_, folder_path_classify) # relDir holds the folder name for each image ie "Elephant"
            relFile = os.path.join(relDir, img_name) # relFile has joining of relDir & img_name "WildPig\Wild_pig_1.jpg"
            relFile = os.path.join(folder_path_classify, relFile) # relFile now holds joining of rootDir and relFile
            # print(relFile)
            if img_name.endswith('.jpg'): # ie "C:/Users/David/Desktop/UPES Research/WRI/images/train/Hare\Hare_194.jpg"
                img_basename = os.path.basename(relFile)
                # print("img_basename", img_basename)
                if not img_basename.startswith('.') and img_basename != 'Thumbs.db':
                    all_image_paths.append(relFile)
                    # print("relFile", relFile)

    # imagePaths = sorted(list(paths.list_images(folder_path_classify)))

    ''' WE CHANGED THIS TO USE NP.LOAD AND TAKE NPZ FILES
    print("trained_text_file from classify: %s" % trained_text_file)
    with open(trained_text_file, "r") as data:
        t = data.read()     # Reads the animals that were used to train a model
        t = t.replace('[', '')
        t = t.replace(']', '')
        t = t.replace("'", "")
        animals_used_to_train = t.split(", ")
        print(animals_used_to_train)
    print(type(animals_used_to_train))
    print("animals used to train this model are: %s" % animals_used_to_train)
    '''

    print(trained_text_file)
    animals_used_to_train = np.load(trained_text_file)
    # animals_used_to_train = ['WildPig', 'BarkingDeer', 'Chital', 'Elephant', 'Gaur', 'Hare', 'Jackal', 'JungleCat', 'Porcupine', 'Sambar', 'SlothBear']
    print(animals_used_to_train)
    t=time.time()
    count = 0


    print("****************************************")


    for animal_dir in animals_used_to_train:
        print(classify_dir_savepath)
        dirName = os.path.join(classify_dir_savepath, animal_dir)
        dirName = dirName.replace("\\","/")
        # print(dirName)
        try:
            # Create target Directory
            os.mkdir(dirName)
            print("Directory ", dirName, " Created ")
        except FileExistsError:
            continue



    for imagePath in all_image_paths:
        basename = os.path.basename(imagePath)
        imagePath = imagePath.replace("\\", "/")
        if not basename.startswith('.') and basename != 'Thumbs.db':
            try:
                print("****************************************")
                print(imagePath)
                # print("Image paths %s" % imagePath)
                count = count +1
                image = cv2.imread(imagePath.replace("\\", "/"))
                orig = image.copy()
                image = cv2.resize(image, (224, 224))
                image = image.astype("float") / 255.0
                image = img_to_array(image)
                image = np.expand_dims(image, axis=0)
                proba = model.predict(image)

                max_proba = np.max(proba)
                label = np.argmax(proba)
                label_name = animals_used_to_train[label]
                print(label_name,proba)
                save_img = os.path.join(classify_dir_savepath, label_name).replace("\\", "/")
                save_img = os.path.join(save_img, basename).replace("\\", "/")
                copyfile(imagePath, save_img)   # Classifying image by saving it in folder which matches the animal

                # print(label_name)

                # label_out = "{}: {:.2f}%".format(label_name, max_proba * 100)
                # output = imutils.resize(orig, width=400)
                # cv2.putText(output, label_out, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # cv2.imwrite('final_output'+str(count)+'.jpg',output)

                #cv2.waitKey(0)
                #cv2.destroyAllWindows()
            except:
                continue

    print('CLASSIFY time: %s' % (t - time.time()))

    print("****************************************")