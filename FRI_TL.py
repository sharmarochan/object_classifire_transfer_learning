# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")
from vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical
from imutils import paths
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
#from GUI_With_Disable import folder_counter
#from GUI_With_Disable import folder_path


# construct the argument parse and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-d", "--dataset", required=True,
# 	help="path to input dataset")
# ap.add_argument("-m", "--model", required=True,
# 	help="path to output model")
# ap.add_argument("-p", "--plot", type=str, default="plot.png",
# 	help="path to output loss/accuracy plot")
# args = vars(ap.parse_args())

# initialize the number of epochs to train for, initial learning rate,
# and batch size

def train_main(folder_counter, folder_path, entry_name, path, used_to_train):
    print("Model name in FRI_TL script is: %s" % entry_name)
    # EPOCHS = 18
    EPOCHS = 1
    BS = 2
    num_classes = folder_counter

    # Initialize the data and labels
    print("[INFO] loading images...")
    data = []
    labels = []
    new_path = folder_path +'/'
    print(new_path)
    # Grab the image paths and randomly shuffle them
    imagePaths = sorted(list(paths.list_images(new_path)))
    print(len(imagePaths))
    random.seed(42)
    random.shuffle(imagePaths)

    count = 0
    rootDir = "D:/Dev_Con_Project/wri_devcon_object_detection/images/train/"
    all_image_paths = []  # An array of all image paths
    animal_subdirectory_names = []  # An array of the names of the different types of animals

    for dir_, _, files in os.walk(rootDir):     # Gets all the image paths for the training of a model
        for img_name in files:
            relDir = os.path.relpath(dir_, rootDir) # relDir holds the folder name for each image ie "Elephant"
            relFile = os.path.join(relDir, img_name) # relFile has joining of relDir & img_name "WildPig\Wild_pig_1.jpg"
            relFile = os.path.join(rootDir, relFile) # relFile now holds joining of rootDir and relFile
            if img_name.endswith('.jpg'): # ie "C:/Users/David/Desktop/UPES Research/WRI/images/train/Hare\Hare_194.jpg"
                all_image_paths.append(relFile)


    for directory, _, files_ in os.walk(rootDir):   # Gets all animal folder names that could be used for training
        animal_folder_name = os.path.relpath(directory, rootDir) # animal_folder_name holds the folder name
        if not animal_folder_name.startswith('.') and animal_folder_name != 'Thumbs.db': # Avoid appending hidden files
            animal_subdirectory_names.append(animal_folder_name)
    animal_subdirectory_names = sorted(animal_subdirectory_names)


    # loop over the input images
    for imagePath in all_image_paths:
        if not imagePath.startswith('.') and imagePath != 'Thumbs.db':
            try:
                new_path = imagePath.replace("\ ", " ")
                # print(new_path)
                # load the image, pre-process it, and store it in the data list
                image = cv2.imread(new_path)
                # cv2.imshow("Output", image)
                # cv2.waitKey(0)
                # print(image)
                image = cv2.resize(image, (224, 224))
                image = img_to_array(image)
                x = np.expand_dims(image, axis=0)
                x = preprocess_input(x)
                data.append(x)
                # extract the class label from the image path and update the
                # labels list
                count = count + 1
                img_folder_name = os.path.basename(os.path.dirname(imagePath))  # Gets name of folder holding the image
                # print("img_folder_name in FRI_TL %s" % img_folder_name)
                for name in animal_subdirectory_names:
                    if name == img_folder_name:     # Gets label for each image and puts it in labels array
                        labels.append(animal_subdirectory_names.index(img_folder_name)) # Gets index of imagefoldername
                print(count)
                if count == 20:
                    break
            except:
                continue

    # scale the raw pixel intensities to the range [0, 1]


    data = np.array(data, dtype="float") / 255.0

    print("[INFO] data matrix: {:.2f}MB".format(data.nbytes / (1024 * 1000.0)))

    print(data.shape)                                   #(100, 1, 224, 224, 3)
    data = np.rollaxis(data,1,0)
    print(data.shape)    							 #(1, 100, 224, 224, 3)
    data = data[0]
    print(data.shape)                                   #(100, 224, 224, 3)

    labels = np.array(labels)
    print(labels.shape)                                 #(100,)
    Y = np_utils.to_categorical(labels, num_classes)        #one hot encoding
    print(Y.shape)                                      #(100, 11)


    print(Y[:5])

    def decode(datum):
        return np.argmax(datum)


    X_train, X_test, y_train, y_test = train_test_split(data, Y, test_size=0.2, random_state=2)

    print("****************************************")

    print(len(X_train))     #80
    print(len(X_test))      #20
    print(len(y_train))     #80
    print(len(y_test))      #20
        
    print("****************************************")

    decoded_y_test = []

    print("****************************************")

    for i in range(0, y_test.shape[0]):
        y = y_test[i]
        decoded_y = decode(y_test[i])
        decoded_y_test.append(decoded_y)
        
    print("****************************************")


    image_input = Input(shape=(224, 224, 3))
    model = VGG16(input_tensor=image_input, include_top=True,weights='imagenet')
    model.summary()

    last_layer = model.get_layer('block5_pool').output
    x = Flatten(name='flatten')(last_layer)
    x = Dense(128, activation='relu', name='fc1')(x)
    x = Dense(128, activation='relu', name='fc2')(x)
    out = Dense(num_classes, activation='softmax', name='output')(x)
    custom_vgg_model2 = Model(image_input, out)
    custom_vgg_model2.summary()

    # freeze all the layers except the dense layers
    for layer in custom_vgg_model2.layers[:-3]:
        layer.trainable = False

    custom_vgg_model2.summary()
    custom_vgg_model2.compile(loss='categorical_crossentropy',optimizer='adadelta',metrics=['accuracy'])


    print("augmentation.....: ")
    aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
        height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
        horizontal_flip=True, fill_mode="nearest")


    t=time.time()
    print("[INFO] training network...")
    #hist = custom_vgg_model2.fit(aug.flow(X_train, y_train, batch_size=32), epochs=EPOCHS, verbose=1, validation_data=(X_test, y_test))
    #hist = custom_vgg_model2.fit_generator(aug.flow(X_train, y_train, batch_size=BS), validation_data=(X_test, y_test), epochs=EPOCHS, verbose=1)
    hist = custom_vgg_model2.fit(X_train, y_train, batch_size=BS, epochs=EPOCHS , verbose=1, validation_data=(X_test, y_test))


    print('Training time: %s' % (t - time.time()))


    print("entry_name from FRI_TL %s" % entry_name)
    print(path+'/'+entry_name+'/'+entry_name+'.model')
    custom_vgg_model2.save(path+'/'+entry_name+'/'+entry_name+'.model')
    # with open(path+'/'+entry_name+'/'+entry_name+'.csv', 'w') as f:     # Writing used_to_train array as text file
    used_to_train = np.array(used_to_train)
    np.save(path+'/'+entry_name+'/'+entry_name, used_to_train) # Writing used_to_train np.array as npz file
    # f.write(str(used_to_train))

    (loss, accuracy) = custom_vgg_model2.evaluate(X_test, y_test, batch_size= BS, verbose=1)

    print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss,accuracy * 100))

    prediction = custom_vgg_model2.predict(X_test,verbose=1)


    decoded_prediction = []
    for i in range(0, prediction.shape[0]):
        y = prediction[i]
        decoded_predic = decode(prediction[i])
        decoded_prediction.append(decoded_predic)
        
    print("****************************************")

    #print(decoded_prediction)           #predicted labels of y_test

    #print(decoded_y_test)               #actual labels of y_test  


    from sklearn.metrics import accuracy_score
    print('Accuracy Score :')
    accuracy = accuracy_score(decoded_y_test, decoded_prediction)
    print(accuracy)


    print('Report : ')
    from sklearn.metrics import classification_report
    print(classification_report(decoded_y_test, decoded_prediction))


    print('Confusion Matrix : ')
    cm = confusion_matrix(decoded_y_test, decoded_prediction)
    print(cm)
    fig = plt.figure(1,figsize=(7,5))
    ax = fig.add_subplot(111)
    cax = ax.matshow(cm)
    plt.title('Confusion matrix of the classifier \n')
    fig.colorbar(cax)
    plt.xlabel('\n Predicted')
    plt.ylabel('True')
    plt.show()
    plt.savefig('animal_final_confusion_Matrix.jpg')


    # visualizing losses and accuracy
    train_loss=hist.history['loss']
    val_loss=hist.history['val_loss']
    train_acc=hist.history['acc']
    val_acc=hist.history['val_acc']

    print(train_loss)

    print("****************************************")

    print(val_loss)

    print("****************************************")

    print(train_acc)

    print("****************************************")

    print(val_acc)

    print("****************************************")
    plt.figure(2,figsize=(7,5))
    plt.plot(train_loss)
    plt.plot(val_loss)
    plt.xlabel('num of Epochs')
    plt.ylabel('loss')
    plt.title('train_loss vs val_loss')
    plt.grid(True)
    plt.legend(['train','val'])
    #print plt.style.available # use bmh, classic,ggplot for big pictures
    plt.style.use('ggplot')
    plt.savefig('animal_final_loss.jpg')

    print("****************************************")

    plt.figure(3,figsize=(7,5))
    plt.plot(train_acc)
    plt.plot(val_acc)
    plt.xlabel('num of Epochs')
    plt.ylabel('accuracy')
    plt.title('train_acc vs val_acc')
    plt.grid(True)
    plt.legend(['train','val'],loc=4)
    #print plt.style.available # use bmh, classic,ggplot for big pictures
    plt.style.use('ggplot')
    plt.savefig('animal_final_accuracy.jpg')

    print("Training Complete")

    '''
    print("****************************************")



    # load the image
    image = cv2.imread('Gaur_1036.jpg')
    orig = image.copy()

    # pre-process the image for classification
    image = cv2.resize(image, (224, 224))
    image = image.astype("float") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)

    # load the trained convolutional neural network
    print("[INFO] loading network...")
    model = load_model('animal_final.model')

    proba = model.predict(image)


    max_proba = np.max(proba)
    label = np.argmax(proba)

    print(max_proba)
    print(label)

    animals = ['WildPig', 'BarkingDeer', 'Chital', 'Elephant', 'Gaur', 'Hare', 'Jackal', 'JungleCat', 'Porcupine', 'Sambar', 'SlothBear']
    label = animals[label]
    print(label)

    label_out = "{}: {:.2f}%".format(label, max_proba * 100)

    # draw the label on the image
    output = imutils.resize(orig, width=400)
    cv2.putText(output, label_out, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # show the output image
    cv2.imshow("Output", output)
    cv2.imwrite('final_output.jpg',output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    '''
