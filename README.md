# cnn-fruits-360
This was a quick try using a Keras CNN for image classification on the fruits 360 dataset provided by Kaggle.

to run on small dataset:

### python3 demo-mini.py

to run on larger dataset:

### python3 demo-full.py


#### Objectives

My main objective for this short project was to train a CNN to classify labelled images of fruits. In the dataset, there were 120 types of fruits so I thought this would be a challenging project based on the sheer number of classes. Also, the original dataset was very large, so I thought it would be excellent to highlight the power of the Nvidia tx2 used.

#### Process

1. Familiarize myself with the file structure. Figure out how to read in data and create the labels from the folder names. Convert png images to numpy arrays of pixels. Reshape data to a tensor with shape (number of images) x (image width) x (image height) x (image depth). I did not rotate or tranform the images because the creators of this dataset already provided multiple images at different angles for each fruit type.

2. For the first CNN model, I chose only two types of fruits to do a binary classification task. In my dense layer, the number of classes was 2 to perform this. I chose bananas and coconuts because they look nothing alike, so I presumed the model could easily classify. It did, with an accuracy of 100%. This is shown in the jupyter notebook titled "keras-binary-classification-initial-model.ipynb".

3. For the second model used, I chose four fruits that look pretty similar because they are all round and either green or red: apple, cherry, avocado, cactus fruit. Again the basic CNN I used ran quickly and had an accuracy of 98.8% when fed the test set. This is shown in the jupyter notebook titled "multiclass-classification-pt1.ipynb". This code also runs with python3 demo-mini.py, and will generate and open a nice .png figure of a random subset of samples with their picture, actual label, and predicted label. This code will run in under 2 minutes.

4. For the third model used, I used a subset of the original dataset from kaggle. The same CNN was trained to predict 120 classes (all the fruits in the dataset). On the test set, the model achieved an accuracy of 81.1%. This is shown in the jupyter notebook titled "multiclass-classification-pt2.ipynb". This code also runs with python3 demo-full.py, and will generate and open a nice .png figure of a random subset of samples with their picture, actual label, and predicted label. This gets interesting, as you can see the model confuse a peeled red onion for an apple or a tomato. 

5. To make the analysis more interesting, I saved png files from google searches to create a new dataset for the model mentioned in step 4. The model did not classify these well. The model trained and tested well on the provided dataset because it was very clean - each image was centered, all were the same size, only one fruit per image, and all background from the photos was removed. Some of the images I added to my new dataset had things in the background, multiple fruits per image, a fruit sliced in half, or a clipart drawing of a fruit. This was the most fun part, however, and if I had more time, I would definitely want to improve my classifier for use on new images.

#### Challenges

The biggest challenge I faced with this project was the size of the dataset. I subset the original kaggle training dataset to between a 1/4 and 1/3 of the size because the tx2 was throwing a low memory error. If I had more time, I would figure out how to read the data incrementally into memory instead of all at once. With the subset of data, the slowest line in the code was normalizing each array value to an integer between 0 and 255.

Another challenge was keeping the run-time of the code around 2 minutes. Granted alot of this was taken up by reading in the data and transforming it, but training the CNN with more than 2-3 epochs typically also took more than 2 minutes. The accuracy for the basic CNN model, was pretty high on the test set, so I did not do a lot of experimenting with model parameters such as batch size or additional Convolution2D layers (or filter size within these layers, 32 seemed to work okay but I could have increase filter size to 64 or 128). The optimizer 'rmsprop' perfomed slightly better than 'adam'. 

#### Results

Using Keras to create a CNN, I was able to achieve an accuracy of 81.1% on a provided test set of images. The CNN classified 120 different types of fruits.

#### Future

If I were to continue this project, I would definitely look into the following: 1) figure out how to read in the data incrementally for modeling, instead of all at once, 2) train the model on a dataset that is not as clean, 3) save my pretrained model to reduce time and improve upon it, and 4) research and implement a model that can identify multiple fruits within one image. I would also use a train, validation, and test set to tweak hyperparameters. Likely, I would research models other than CNN after creating a more robust and sophisiticated CNN. Transfer learning, for example could be useful and I saw some scholarly articles on VGG and ResNet. 

