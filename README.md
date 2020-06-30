# Image Captioning
___
Implementation of a Image Captioning Model using Transfer Learning and Recurrent Neural Networks

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/iVishalr/Image-Captioning/blob/master/Notebook/ImageCaptioning.ipynb)

## Requirements
___
The following libraries are required to run the code : 
  1. Tensorflow
  2. Keras
  3. Numpy
  4. Pandas
  5. OpenCV
  6. Matplotlib
  7. Tqdm

The above mentioned packages can be installed by executing ```pip install -r requirements.txt```

### Dataset
___

The neural network has been trained on Flicker8K Dataset. This dataset has been obtained from Kaggle. The link to this dataset is given below. Please download the dataset from the given link and place them in your project directory.

Link : https://www.kaggle.com/ming666/flicker8k-dataset

### Training
___

Training the neural network takes a really long time. Training depends on your system's capability. The neural network has been trained on Nvidia Tesla K80 GPUs for about 3 days. This could vary from system to system. It is recommended to use a GPU of some sort to train this model.

I would like to thank Kaggle for letting me use their GPUs for training my neural network.

### Contribution
___

Contributers are free to make changes in the code either to imporve it or correct some of the bugs that may be present in the code. Please make a pull request before you edit the code. Avoid making changes in the master branch.

For downloading the weights for the Neural Network, please contact me at my email address given below

Email ID : *vishalramesh01[at]gmail.com*

*NOTE : The model weights will not be given to everyone. The chances of getting one will be based on your contributions to the       project.*

### Results
___

![alt person](https://github.com/iVishalr/Image-Captioning/blob/master/images/person.jpeg?raw=true) 

Generated Caption : *A man is standing on a snow covered mountain.*

![alt dog](https://github.com/iVishalr/Image-Captioning/blob/master/images/dog.jpeg?raw=true)

Generated Caption : *A brown dog is running towards the camera with a stick in its mouth.*

### References
___

The following resources were utilised in developing this project.

1. MIT Deep Learning Basics: Introduction and Overview
   
   https://youtu.be/O5xeyoRL95U
2. CS231n Winter - 2016, Stanford Univeristy
   Lecture - 10 : Reccurent Neural Networks and Image Captioning, LSTM by Andrej Karpathy.
   
   https://youtu.be/yCC09vCHzF8
3. Training Neural Networks, Stanford University
   
   https://youtu.be/wEoyxE0GP2M
4. Andrej Karpathy's Neuraltalk2
   
   https://github.com/karpathy/neuraltalk2
5. Automated Image Captioning using ConvNets and Reccurent Nets
   By Andrej Karpathy and Fei - Fei Li
   
   https://cs.stanford.edu/people/karpathy/sfmltalk.pdf
