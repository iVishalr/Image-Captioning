#!/usr/bin/env python
# coding: utf-8
import numpy as np
import os
import string
from PIL import Image
import pickle
import cv2
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Model, load_model
from keras.layers.merge import add
from keras.utils import to_categorical
from keras.layers import Dense, Flatten,Input, Convolution2D, Dropout, LSTM, TimeDistributed, Embedding, Bidirectional, Activation, RepeatVector,Add
from keras.applications import InceptionV3
from keras.preprocessing.image import load_img, img_to_array
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm_notebook as tqdm
from keras.preprocessing import image
from keras.optimizers import RMSprop

def load_document(filename):
  file = open(filename,'r')
  text = file.read()
  file.close()
  return text

def generate_img_captions(filename):
  file = load_document("Flickr8k_text/Flickr8k.token.txt")
  captions = file.split('\n')
  descriptions ={}
  for caption in captions[:-1]:
    img, caption = caption.split('\t')
    if img[:-2] not in descriptions:
      descriptions[img[:-2]] = []
    descriptions[img[:-2]].append(caption)
  return descriptions

def save_descriptions(descriptions,filename):
    lines = list()
    for key,caption_list in descriptions.items():
        for caption in caption_list:
            lines.append(key + '\t' + caption)
    image_descriptions = '\n'.join(lines)
    file = open(filename,'w')
    file.write(image_descriptions)
    file.close()

dataset_images = "Flickr8k_Dataset/Flicker8k_Dataset"
dataset_text = "Flickr8k_text"
filename = dataset_text + "/Flickr8k.token.txt"
descriptions = generate_img_captions(filename)
print(len(descriptions))
save_descriptions(descriptions,"descriptions.txt")

train_dataset = "Flickr8k_text/Flickr_8k.trainImages.txt"
train_data_img = open(train_dataset).read().split("\n")

model = InceptionV3(weights="imagenet")
new_input = model.input
hidden_layer = model.layers[-2].output
model_inc = Model(new_input,hidden_layer)
model_inc.summary()
#Pre-processing the images
def preprocess_input(x):
    x /= 255.
    x -= 0.5
    x *= 2.
    return x
def preprocess(image_path):
    img = load_img(image_path, target_size=(299, 299))
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

def encode(image):
    image = preprocess(image)
    temp_enc = model_vgg.predict(image)
    temp_enc = np.reshape(temp_enc, temp_enc.shape[1])
    return temp_enc
#Execute the following lines of code only for the first time.
features = {}
for img in train_data_img:
    if(len(img)>0):
        features[img] = encode("Flickr8k_Dataset/Flicker8k_Dataset/"+img)

#Store the extracted features
with open("Features.p", "wb") as f:
    pickle.dump(features, f)

#Loading the feautres extracted from enode()
features = pickle.load(open("Features.p","rb"))

training_descriptions = {}
for image in descriptions:
    if(image in train_data_img):
        training_descriptions[image] = descriptions[image]

#Calculating vocabulary
caps = []
for key, val in training_descriptions.items():
    for i in val:
        caps.append('<start> ' + i + ' <end>')

#This section is to be executed only once. Please load in unique.p if you want to run more than once.
words = [i.split() for i in caps]
unique = []
for i in words:
    unique.extend(i)
unique = list(set(unique))
with open("unique.p", "wb") as pickle_d:
    pickle.dump(unique, pickle_d)

#Mapping the words in unique to a particular index
word2index = {val:index for index, val in enumerate(unique)}
index2word = {index:val for index, val in enumerate(unique)}

#Calculating maximum length
maximum_length = 0
for c in caps:
    c = c.split()
    if len(c) > maximum_length:
        maximum_length = len(c)

#Calculate Vocabulary Size
vocabulary_size = len(unique)

f = open('flickr8k_training_dataset.txt', 'w')
f.write("image_id\tcaptions\n")
for key, val in training_descriptions.items():
    for i in val:
        f.write(key + "\t" + "<start> " + i +" <end>" + "\n")
f.close()
df = pd.read_csv('flickr8k_training_dataset.txt', delimiter='\t')
samples_per_epoch = 0
for ca in caps:
    samples_per_epoch += len(ca.split())-1
def data_generator(batch_size = 32):
        partial_caps = []
        next_words = []
        images = []
        
        df = pd.read_csv('flickr8k_training_dataset.txt', delimiter='\t')
        df = df.sample(frac=1)
        iter = df.iterrows()
        c = []
        imgs = []
        for i in range(df.shape[0]):
            x = next(iter)
            c.append(x[1][1])
            imgs.append(x[1][0])
        count = 0
        while True:
            for j, text in enumerate(c):
                current_image = features[imgs[j]]
                for i in range(len(text.split())-1):
                    count+=1
                    
                    partial = [word2index[txt] for txt in text.split()[:i+1]]
                    partial_caps.append(partial)
                    
                    # Initializing with zeros to create a one-hot encoding matrix
                    # This is what we have to predict
                    # Hence initializing it with vocab_size length
                    n = np.zeros(vocabulary_size)
                    # Setting the next word to 1 in the one-hot encoded matrix
                    n[word2index[text.split()[i+1]]] = 1
                    next_words.append(n)
                    
                    images.append(current_image)

                    if count>=batch_size:
                        next_words = np.asarray(next_words)
                        images = np.asarray(images)
                        partial_caps = pad_sequences(partial_caps, maxlen=maximum_length, padding='post')
                        yield [[images, partial_caps], next_words]
                        partial_caps = []
                        next_words = []
                        images = []
                        count = 0

#Model Design
embedding_size=300

image_model = Input(shape=(2048,))
image_model1 = Dense(embedding_size, activation='relu')(image_model)
image_model2 = RepeatVector(maximum_length)(image_model1) 

language_model = Input(shape=(maximum_length,))
seq1 = Embedding(vocabulary_size,embedding_size,mask_zero=True)(language_model)
seq2 = LSTM(256, return_sequences=True)(seq1)
seq3 = TimeDistributed(Dense(embedding_size))(seq2)

decoder1 = add([image_model2, seq3])
decoder2 = Bidirectional(LSTM(256,return_sequences=False))(decoder1)
outputs1 = Dense(vocabulary_size)(decoder2)
outputs = Activation("softmax")(outputs1)
model = Model(inputs=[image_model,language_model],outputs=outputs)
model.compile(loss='categorical_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])
model.summary()
#Training
j=1 #update this after every 'x' epochs
x=10
for i in range(x):
    model.fit_generator(data_generator(batch_size=128), steps_per_epoch=samples_per_epoch/128, nb_epoch=1, verbose=1)
model.save("Model-InceptionV"+str(j)+".h5")

def predict_captions(image):
    start_word = ["<start>"]
    while True:
        par_caps = [word2index[i] for i in start_word]
        par_caps = pad_sequences([par_caps], maxlen=maximum_length, padding='post')
        e = encode(image)
        preds = model.predict([np.array([e]), np.array(par_caps)])
        word_pred = index2word[np.argmax(preds[0])]
        start_word.append(word_pred)
        
        if word_pred == "<end>" or len(start_word) > maximum_length:
            break
            
    return ' '.join(start_word[1:-1])

def beam_search_predictions(image, beam_index = 3):
    start = [word2index["<start>"]]
    
    start_word = [[start, 0.0]]
    
    while len(start_word[0][0]) < maximum_length:
        temp = []
        for s in start_word:
            par_caps = pad_sequences([s[0]], maxlen=maximum_length, padding='post')
            e = encode(image)
            preds = model.predict([np.array([e]),np.array(par_caps)])
            
            word_preds = np.argsort(preds[0])[-beam_index:]
            
            # Getting the top <beam_index>(n) predictions and creating a 
            # new list so as to put them via the model again
            for w in word_preds:
                next_cap, prob = s[0][:], s[1]
                next_cap.append(w)
                prob += preds[0][w]
                temp.append([next_cap, prob])
                    
        start_word = temp
        # Sorting according to the probabilities
        start_word = sorted(start_word, reverse=False, key=lambda l: l[1])
        # Getting the top words
        start_word = start_word[-beam_index:]
    
    start_word = start_word[-1][0]
    intermediate_caption = [index2word[i] for i in start_word]

    final_caption = []
    
    for i in intermediate_caption:
        if i != '<end>':
            final_caption.append(i)
        else:
            break
    
    final_caption = ' '.join(final_caption[1:])
    return final_caption

try_image = "images/dog.jpeg"
Image.open(try_image)

print ('Normal Max search:', predict_captions(try_image)) 
print ('Beam Search, k=3:', beam_search_predictions(try_image, beam_index=3))
print ('Beam Search, k=5:', beam_search_predictions(try_image, beam_index=5))
print ('Beam Search, k=7:', beam_search_predictions(try_image, beam_index=7))
