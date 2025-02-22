#Dataset from Kaggle
#1. Import libraries
import glob
#import keras
import os
import zipfile
from tokenize import group

import cv2 #gives access to all the functions and classes that OpenCV offers for image processing, computer vision, and machine learning tasks
import gdown
import h5py
import pandas as pd
import pip
#from cycler import K
#from keras.model import Sequential
#from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import tensorflow.compat.v1 as tf
from contourpy.util import data
from pygments.lexers import python
from tensorflow.keras.utils import to_categorical
import tensorflow as tf; print(tf.__version__)
#in terminal add pip install tensorflow-addons
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dropout
from tensorflow.keras.regularizers import l2

from keras import Input, Model
from keras.src.callbacks import EarlyStopping, ModelCheckpoint
from keras.src.layers import Conv2D, MaxPooling2D, SeparableConv2D, BatchNormalization, Flatten, Dense, Dropout
from keras.src.optimizers import Adam
from keras.src.utils import to_categorical #Converts a class vector (integers) to binary class matrix
from numpy import append
from tensorflow._api.v2 import train
from tensorflow.python.client import session
#import imgaug as aug
#pip install imgaug
import albumentations as A #need to pip install albumentations in terminal
from albumentations.core.composition import OneOf
from albumentations.augmentations.geometric.transforms import HorizontalFlip, ShiftScaleRotate
from albumentations.augmentations.transforms import RandomBrightnessContrast
import random
import seaborn as sns
from pathlib import Path
import matplotlib.pyplot as plt
from skimage.io import imread #simplifies loading images in a way that they can be directly processed or analyzed using Python.


#from tensorflow.python.client import session


#2. Downloading/extracting dataset
def extract_zip(zip_filepath, extract_folder):
    # Ensure the folder exists
    os.makedirs(extract_folder, exist_ok=True)

    # Extract the contents of the zip file
    with zipfile.ZipFile(zip_filepath, 'r') as zip_ref:
        zip_ref.extractall(extract_folder)
    print(f"Extracted files to {extract_folder}.")

if not os.path.exists("kaggle_data"):
    os.mkdir("kaggle_data")
    # Extract the zip file contents
    extract_zip("kaggle_data.zip", "kaggle_data")

#3. Data Preparation
#To make the results reproducible seeding is done

#set the seed for hashbased operations-when running multiple experiments to ensure consistent results
os.environ['PYTHONHASHSEED'] = '0'

#setting global seed for randomness
SEED = 111
random.seed(SEED)
np.random.seed(SEED)

#disable multithreading in tensorflow ops helps avoid non-deterministic behavior caused by parallel execution.

session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads = 1)

#set the random seed in tensorflow at graph level-Sets the random seed for TensorFlow operations (e.g., initializing weights, shuffling data)
tf.compat.v1.set_random_seed(111)

#define a tensor flow session with above session configs
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config= session_conf)

#set the session in keras
# K.set_session(sess) #not needed in TF version 2

#make the augmentation sequence deterministic
#A.seed(111)

#4. Dataset is divided into train, test and val. Let's see the data
data_dir = Path('./kaggle_data/chest_xray/chest_xray')

print(type(data_dir))
#path to train directory
train_dir = data_dir / 'train'

#path to test directory
test_dir = data_dir/'test'

#path to validation directory
val_dir = data_dir/ 'val'

'''
We will first go through the training dataset. We will do some analysis on that, look at some of the samples, check the number of samples for each class, etc. Lets' do it.

Each of the above directory contains two sub-directories:

NORMAL: These are the samples that describe the normal (no pneumonia) case.
PNEUMONIA: This directory contains those samples that are the pneumonia cases.
'''

#getting path to normal and pneumonia sub-directories for training data
normal_cases_dir = train_dir/ 'Normal'
pneumonia_cases_dir = train_dir/ 'Pneumonia'

#Get the list of all images
#glob('*.jpeg') retrieves an iterator over all files in the directory ending with .jpeg.
normal_cases = normal_cases_dir.glob('*.jpeg')
pneumonia_cases = pneumonia_cases_dir.glob('*.jpeg')

#an empty list. we will insert the data into it
train_data = []

#Go through all normal cases.The label of these will be 0
for img in normal_cases:
    train_data.append((img,0))

#Go through all pneumonia cases. The label of these will be 1
for img in pneumonia_cases:
    train_data.append((img,1))

#get a pandas data frame from the data we have in our list
train_data = pd.DataFrame(train_data, columns=['image', 'labels'], index=None)

#Shuffle the data
##frac=1.0 means that 100% of the rows are included in the sampling.
#reset_index(drop=True): Resets the index of the shuffled DataFrame and removes the old index (drop=True ensures the old index is not added as a new column).
train_data = train_data.sample(frac=1.).reset_index(drop=True)

#how the dataframe looks like

print(train_data.shape)

#get the count of each class
cases_count = train_data['labels'].value_counts()
print(cases_count)

#plot the results
plt.figure(figsize=(10,8))
sns.barplot(x=cases_count.index, y = cases_count.values)
plt.title('Number of cases', fontsize=14)
plt.xlabel('Case Type', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.xticks(range(len(cases_count.index)), ['Normal(0)', 'Pneumonia(1)'])
plt.show()
#data is imbalanced, with thrice as many cases of pneumonia as normal

#Lets see how penumonia cases different from normal cases
pneumonia_samples = (train_data[train_data['labels']==1]['image'].iloc[:5]).tolist()
normal_samples = (train_data[train_data['labels']==0]['image'].iloc[:5]).tolist()

#concat the data in single list and deleting the lists
samples = pneumonia_samples + normal_samples
del pneumonia_samples, normal_samples

#plot the data
f, ax = plt.subplots(2,5, figsize=(30,10))
for i in range(10):
    img = imread(samples[i])
    ax[i//5, i%5].imshow(img, cmap = 'gray')
    if i<5:
        ax[i//5, i%5].set_title("Pneumonia")
    else:
        ax[i//5, i%5].set_title("Normal")
    ax[i//5, i%5].axis('off')
    ax[i//5, i%5].set_aspect('auto')
plt.show()

#5. Preparaing Validation data
#get the path to sub-directories
normal_cases_dir = val_dir/'NORMAL'
pneumonia_cases_dir = val_dir/'PNEUMONIA'

#Get the list of all the images
normal_cases = normal_cases_dir.glob('*.jpeg')
pneumonia_cases = pneumonia_cases_dir.glob('*.jpeg')

# List that are going to contain validation images data and the corresponding labels
valid_data = []
valid_labels = []

#Normalizing pixel value and converting all images to RBG from grayscale to a size of 224*224

#Normal Cases
for img in normal_cases:
    img = cv2.imread(str(img))
    img = cv2.resize(img, (224,224))
    if img.shape[2] ==1:
        img = np.dstack([img,img,img])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)/255.
    label = to_categorical(0, num_classes=2)
    valid_data.append(img)
    valid_labels.append(label)


#Pneumonia cases
for img in pneumonia_cases:
    img = cv2.imread(str(img))
    img = cv2.resize(img, (224,224))
    if img.shape[2] ==1:
        img = np.dstack([img,img,img])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)/225.
    label = to_categorical(1, num_classes=2)
    valid_data.append(img)
    valid_labels.append(label)

#converting list to numpy array

valid_data = np.array(valid_data)
valid_labels = np.array(valid_labels)

print("Total number of valid samples", valid_data.shape)
print("Total number of labels", valid_labels.shape)


#6. Data Augmentation

#Is of great help if dataset is imbalanced. Can help to generate different samples of undersampled class.
transform = A.OneOf([
    A.HorizontalFlip(p=0.5), #horizontal flip
    A.Rotate(limit=20, p=0.5), #rotation with a 20degree
    A.RandomBrightnessContrast(p=0.5) #random brightness and contract adjustment
], p=1.0) #overall probability of one

#6.1 Training data generator
"""
The code defines a function data_gen to create a data generator for 
training deep learning models. It prepares batches of images and their 
corresponding labels, handling class imbalance by augmenting images from the undersampled class.

"""

def data_gen(data, batch_size, class_weight):
    """
    A generator that yields (x, y, sample_weight) batches from a DataFrame.

    Args:
    data: Pandas DataFrame with columns 'image' (file path) and 'labels' (class).
    batch_size: Number of samples per batch.
    class_weight: Dictionary of class weights, e.g., {0: 1.0, 1: 0.4}.

    Yields:
    batch_data: Batch of input images (NumPy array).
    batch_labels: Batch of one-hot encoded labels (NumPy array).
    sample_weight: Batch of sample weights (NumPy array).
    """
    #get total number of samples in the data
    n = len(data)
    steps = n//batch_size

    #Define two numpy arrays for containing batch data and labels
    batch_data = np.zeros((batch_size, 224, 224, 3), dtype = np.float32)
    batch_labels = np.zeros((batch_size, 2), dtype=np.float32)
    batch_weights = np.zeros(batch_size, dtype=np.float32) #array of sample weights

    #get a numpy array of all the indices of th input data
    indices = np.arange(n)

    #intilialize a counter
    i=0 #intialize the batch index
    while True:   #is a data generator loop, typically used for preparing batches of images and labels during training in machine learning models.
        np.random.shuffle(indices)
        #get the next batch
        count = 0
        next_batch = indices[(i*batch_size): (i+1)*batch_size]

        for j, idx in enumerate(next_batch):
            img_name = data.iloc[idx]['image']
            label = data.iloc[idx]['labels']

            #one hot encoding
            encoded_label = to_categorical(label, num_classes=2)

            #read the img and resize  #ensures compatibility with the model architecture.
            img = cv2.imread(str(img_name))
            img = cv2.resize(img, (224,224))

            #check if its grascale
            if img.shape[2] ==1:
                img= np.dstack([img, img, img])

            #cv2 reads in BGR code
            orig_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            #Normalize the image pixels
            orig_img = orig_img.astype(np.float32)/225.

            batch_data[count] = orig_img
            batch_labels[count] = encoded_label
            batch_weights[count] = class_weight[label] #add sample weights based on class weights

            #generating more samples of undersamples class
            if label ==0 and count < batch_size-2:
                a_img1 = transform(image=img)['image']
                a_img2 = transform(image=img)['image']
                a_img1 = cv2.cvtColor(a_img1, cv2.COLOR_BGR2RGB).astype(np.float32)/225.
                a_img2 = cv2.cvtColor(a_img2, cv2.COLOR_BGR2RGB).astype(np.float32)/225.


                batch_data[count+1] = a_img1 #count+1manages indices properly so no data is overwritten in the batch arrays.
                batch_labels[count+1] = encoded_label
                batch_weights[count+1] = class_weight[label]


                batch_data[count+2] = a_img2
                batch_labels[count+2] = encoded_label
                batch_weights[count+2] = class_weight[label]

                count+=2 #After adding two images and labels, the counter (count) is incremented by 2 to keep track of the current position in the batch arrays.

            #stop when the batch is full
            else:
                count+=1 #If there is no augmentation or additional image processing required, the code increments the count by 1 to prepare for the next image in the loop.

            if count==batch_size:
                break #Breaking the loop when count == batch_size - 1

        i+=1
        yield batch_data, batch_labels, batch_weights


#reset batch index after completing one epoch
        if 1>=steps:
            i=0

#7. Building model
"""
NOTES- There will be partial transfer learning and rest of the model will be trained from scratch.
Choose a simple architecture.
1. Initialize the first few layers from a network that is pretrained on imagenet. 
as it capture general details like color blobs, patches, edges, etc. 
2.Instead of randomly initialized weights for these layers, it would be much better if you fine tune them.
3. Choose layers that introduce a lesser number of parameters. 
For example, Depthwise SeparableConv is a good replacement for Conv layer. It introduces lesser number of 
parameters as compared to normal convolution and as different filters are applied to each channel,
 it captures more information. Xception a powerful network, is built on top of such layers only.
4. Use batch norm with convolutions for the network that becomes deeper.
5. Add dense layers with reasonable amount of neurons. Train with a higher learning rate.
Do it for the depth of your network too.
6. Once you know a good depth, start training your network with a lower learning rate along with decay.

"""
def build_model(input_image=None):
        input_img = Input(shape=(224,224,3), name = 'ImageInput')
        x = Conv2D(64, (3,3), activation='relu', padding='same', kernel_regularizer=l2(0.001), name= 'Conv1_1')(input_img)
        x = Conv2D(64, (3,3), activation='relu', padding='same', kernel_regularizer=l2(0.001), name = 'Conv1_2')(x)
        x = MaxPooling2D((2,2), name = 'pool1')(x)

        x = SeparableConv2D(128, (3,3), activation='relu', padding='same',name = 'Conv2_1')(x)
        x = BatchNormalization(name='bn1_1')(x)

        x = SeparableConv2D(128,(3,3), activation='relu', padding='same', name = 'Conv2_2')(x)
        x = BatchNormalization(name='bn2_2')(x)

        x = MaxPooling2D((2,2), name = 'pool2')(x)

        x = SeparableConv2D(256, (3,3), activation='relu', padding = 'same', name = 'Conv3_1')(x)
        x = BatchNormalization(name = 'bn1')(x)

        x = SeparableConv2D(256, (3,3), activation='relu', padding='same', name = 'Conv3_2')(x)
        x= BatchNormalization(name = 'bn2')(x)

        x = SeparableConv2D(256, (3,3), activation = 'relu', padding = 'same', name = 'Conv3_3')(x)
        x = MaxPooling2D((2,2), name = 'pool3')(x)

        x = SeparableConv2D(512, (3,3), activation='relu', padding = 'same', name = 'Conv4_1')(x)
        x = BatchNormalization(name = 'bn3')(x)

        x = SeparableConv2D(512, (3,3), activation = 'relu', padding = 'same', name = 'Conv4_2')(x)
        x = BatchNormalization(name = 'bn4')(x)

        x = SeparableConv2D( 512, (3,3), activation = 'relu', padding = 'same', name = 'Conv_4_3')(x)
        x = MaxPooling2D((2,2), name = 'pool4')(x)

        x = Flatten(name = 'flatten')(x)
        x = Dense(1024, activation = 'relu', kernel_regularizer=l2(0.001), name = 'fc1')(x)
        x = Dropout(0.7, name = 'dropout1')(x)
        x = Dense(512, activation = 'relu', kernel_regularizer=l2(0.001), name = 'fc2')(x)
        x = Dropout(0.5, name = 'dropout2')(x)
        x = Dense(2, activation = 'softmax', name = 'fc3')(x)

        model = Model( inputs=input_img, outputs=x)
        return model

model = build_model()
model.summary()

#8 Initializing the weights for first two convolutions with imagenet

#open the VGG16 weight file

"""
h5py stands for  the h5py library in Python, which is used to read and write HDF5 
(Hierarchical Data Format version 5) files. HDF5 is a data model, library, and file format 
designed to store and organize large amounts of data efficiently.

Machine Learning: Saving and loading datasets, model weights, or intermediate results.
Scientific Computing: Handling large numerical datasets (e.g., satellite images, simulations).
Bioinformatics: Storing genome sequences or experimental data.

"""
f = h5py.File('/Users/bhavna/Desktop/My_Computer/Deep_Learning_Specialisation/Practice_projects/New_project_XRay_images/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5', 'r')

#select the layers for which you want to set weights

#standard conv2d
w, b = f['block1_conv1']['block1_conv1_W_1:0'],f['block1_conv1']['block1_conv1_b_1:0']
model.layers[1].set_weights = [w,b]

#standard conv2d
w, b = f['block1_conv2']['block1_conv2_W_1:0'], f['block1_conv2']['block1_conv2_b_1:0']
model.layers[2].set_weights = [w,b]

#separable conv2d
w, b = f['block2_conv1']['block2_conv1_W_1:0'],f['block2_conv1']['block2_conv1_b_1:0']
model.layers[4].set_weights= [w,b]

#separable conv2d
w, b = f['block2_conv2']['block2_conv2_W_1:0'],f['block2_conv2']['block2_conv2_b_1:0']
model.layers[5].set_weights = [w, b]

f.close()
model.summary()

#9. Optimization
#purpose is finding the optimal parameters that minimize the loss function and quantifying how well

batch_size= 16
nb_epochs = 20

#define the number of training steps
nb_train_steps = train_data.shape[0]//batch_size

# the model predicts the target variable
lr_schedule = ExponentialDecay(initial_learning_rate=3e-4, decay_steps= 20000,decay_rate=0.98 , staircase=True)
opt = Adam(learning_rate=lr_schedule)
es = EarlyStopping(
    monitor= 'val_loss',
    patience=10,
    restore_best_weights=True
)
chkpt = ModelCheckpoint(filepath = 'best_model_todate.weights.keras',
                        save_best_only = True,
                        save_weights_only = False # Save full model not just the weights
                        )


#using focal loss for down weighting the imbalanced data

model.compile(
    loss= 'binary_crossentropy',
    metrics = ['accuracy'],
    optimizer = opt
)

#get a train generator data
class_weight = {0:1.0, 1:0.4} #adjust weights bacsed onclass distribution

train_data_gen = data_gen(train_data, batch_size = batch_size, class_weight = class_weight)

print("Number of training and validation steps: {} and {}". format(nb_train_steps, len(valid_data)))

#Fit the model
#Have put fit model command in estrics as do not want it to run every time
"""history = model.fit(train_data_gen,
                    epochs=nb_epochs,
                    steps_per_epoch=nb_train_steps,
                    validation_data = (valid_data, valid_labels),
                    callbacks=[es, chkpt] #use early stopping and model checkpoint
                              )
"""
#load weights
model.load_weights("/Users/bhavna/Desktop/My_Computer/Deep_Learning_Specialisation/Practice_projects/New_project_XRay_images/best_model_todate.weights.h5")

#preparing test data
normal_cases_dir = test_dir/'NORMAL'
pneumonia_case_dir = test_dir/'PNEUMONIA'

normal_cases = normal_cases_dir.glob('*.jpeg')
pneumonia_cases = pneumonia_cases_dir.glob('*.jpeg')

test_data = []
test_labels = []

for img in normal_cases:
    img = cv2.imread(str(img))
    img = cv2.resize(img,(224,224))
    if img.shape[2]==1:
        img = np.dstack([img,img,img])
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)/255
    label = to_categorical(0, num_classes=2)
    test_data.append(img)
    test_labels.append(label)

for img in pneumonia_cases:
    img = cv2.imread(str(img))
    img = cv2.resize(img, (224,224))
    if img.shape[2]==1:
        img = np.dstack([img,img,img])
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)/255
    label = to_categorical(1, num_classes=1)
    test_data.append(img)
    test_labels.append(label)

test_data = np.array(test_data)
test_labels = np.array(test_labels)

print("Total number of Training examples: ",test_data.shape)
print("Total number of labels: ", test_labels.shape)

#Evaluation of test dataset

test_loss, test_score = model.evaluate(test_data, test_labels, batch_size=16)
print("Loss on test set: ", test_loss)
print("Accuracy of test score: ", test_score)

#get predictions
preds = model.predict(test_data, batch_size=16)
preds = np.argmax(preds, axis=-1)

#original labels
orig_test_labels = np.argmax(lest_labels, axis=-1)

print(orig_test_labels.shape)
print(preds.shape)

#Get confusion matrix
cm = confusion_matrix(orig_test_labels, pred)
plt.figure()
plot_confusion_matrix(cm, figsize=(12,8), hide_ticks=True, alpha=0.7, cmap=plt.cm.Blues)
plt.xticks(range(2), ['Normal', 'Pneumonia'], fontsize=16)
plt.yticks(range(2), ['Normal', 'Pneumonia'], fontsize=16)
plt.show()

#calculate precision and recall
tn, fp, fn, tp = cm.ravel()

precision = tp/(tp+fp)
recall = tp/(tp+fn)

print("Recall of the model is  {:.2f}".format(recall))
print("Precision of the model is {:.2f}".format(precision))





















