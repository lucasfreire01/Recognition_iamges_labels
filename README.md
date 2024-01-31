# Recognition_iamges_labels
Hello guys I'm here again, to show one more project this time using the Google Net (inception) architecture to predict what the label respective.
### GoogleNet(Inception)
GoogleNet, also known as Inception, is a groundbreaking deep convolutional neural network architecture developed by Google. Introduced in 2014, this innovative model revolutionized the field of computer vision with its unique inception modules. GoogleNet achieved state-of-the-art performance in image classification tasks, demonstrating both remarkable accuracy and computational efficiency.

What sets GoogleNet apart is its utilization of inception modules, which employ multiple filter sizes within the same layer to capture diverse features at different scales. This allows the model to learn and represent intricate patterns in images effectively. Moreover, the architecture incorporates global average pooling, reducing the number of parameters and mitigating overfitting, contributing to its efficiency.

GoogleNet's success not only lies in its superior performance but also in its ability to balance accuracy and computational resources, making it a pivotal contribution to the evolution of deep learning architectures. This model has inspired subsequent advancements in neural network design and remains influential in the development of state-of-the-art models for image recognition tasks.

## EDA
The train and test dataset is load by keras.datasets.cifar10.load_data()<br>.
we split automatic in x_train, x_test, y_train, y_test. Each one have images but variables is showed like this for us:
array([[[[ 59,  62,  63],<br>
         [ 43,  46,  45],<br>
         [ 50,  48,  43],<br>
         ...,<br>
         [158, 132, 108],<br>
         [152, 125, 102],<br>
         [148, 124, 103]],<br>
<br>
        [[ 16,  20,  20],<br>
         [  0,   0,   0],<br>
         [ 18,   8,   0],<br>
         ...,<br>
         [123,  88,  55],<br>
         [119,  83,  50],<br>
         [122,  87,  57]],<br>
<br>
        [[ 25,  24,  21],<br>
         [ 16,   7,   0],<br>
         [ 49,  27,   8],<br>
         ...,<br>
         [118,  84,  50],<br>
         [120,  84,  50],<br>
         [109,  73,  42]],<br>
<br>
        ...,<br>
All datasets have this matrix each set of cords represents an image and each cord with 3 values(RGB) in the training dataset we have 50000 values, the X_train we get the shape(50000, 32, 32, 3) and in the test datasets we get 10000 values the x_test have the shape (10000, 32, 32, 3) the rest follow this rule. This is an example of an image in RGB<br>

![colorful_picture](https://github.com/lucasfreire01/Recognition_iamges_labels/blob/main/reports/figures/figure(1).png)<br>
Bellow there is an example of the images and your labels.<br>
![label_picture](https://github.com/lucasfreire01/Recognition_iamges_labels/blob/main/reports/figures/figure(2).png)<br>

## Pre_Processing
### Type and grayscale
The first thing did was transform the type of values from int to float32 because processing is more computationally cheap, normalization of these variables based in white(255)next used coding in the y_train and y_test and transformed in the grayscale, because images are 3 buses (R, G, B) for example (187, 222, 87) when we get a gray image in RGB, will be for example(187, 187, 187) we have gray in the RGB base when all values are the same the color is gray this enables us to use 1 buses because the number is the same that is the 3 buses(R G B), bellow is an example in the gray. After we norm these values based on the white number in RGB (255). Here there is an example of grayscale.<br>
![colorful_picture](https://github.com/lucasfreire01/Recognition_iamges_labels/blob/main/reports/figures/figure(3).png)<br>
![gray_picture](https://github.com/lucasfreire01/Recognition_iamges_labels/blob/main/reports/figures/figure(4).png)<br>
### Noise_Random
We dds random noise to a normalized grayscale training dataset (x_train_norm_gray). The noise is generated using a normal distribution with a standard deviation of 0.1. The purpose is to introduce variability into the data, aiding the model in learning to be more robust by adapting to different patterns and reducing overfitting. This technique, known as data augmentation, enhances the model's generalization capabilities without requiring additional labeled data.<br>
![colorful_picture](https://github.com/lucasfreire01/Recognition_iamges_labels/blob/main/reports/figures/figure(5).png)<br>
![colorful_picture](https://github.com/lucasfreire01/Recognition_iamges_labels/blob/main/reports/figures/figure(6).png)<br>

## Model
### Architecture
How say we use Google Net bellow is a picture of the architecture:
![colorful_picture](https://github.com/lucasfreire01/Recognition_iamges_labels/blob/main/reports/arquitecture/GoogleNet.png)<br>
This architecture is inception with basic with a convolution 1x1, 3x3, and 5x5 and this case uses the max pooling, other max pooling 1x1 and concatenates these variables, we concept the neural networking processing and we get this summary:
| Layer                  | Output Shape              | Param # | Connected to               |
|------------------------|---------------------------|---------|----------------------------|
| input_2 (InputLayer)   | [(None, 224, 224, 3)]     | 0       |                            |
| max_pooling2d_4         | (None, 224, 224, 3)       | 0       | input_2[0][0]              |
| conv2d_9               | (None, 224, 224, 64)      | 256     | input_2[0][0]              |
| conv2d_10              | (None, 224, 224, 128)     | 3584    | input_2[0][0]              |
| conv2d_11              | (None, 224, 224, 32)      | 2432    | input_2[0][0]              |
| conv2d_12              | (None, 224, 224, 32)      | 128     | max_pooling2d_4[0][0]      |
| concatenate_2          | (None, 224, 224, 256)     | 0       | conv2d_9[0][0],            |
|                        |                           |         | conv2d_10[0][0],           |
|                        |                           |         | conv2d_11[0][0],           |
|                        |                           |         | conv2d_12[0][0]            |

After we build the architecture of neural networking final, using Conv2d as before but this time using maxpooling2d, three blocks of these layers in the last we have  the max pooling but also average global pooling to try to reduce more the dimensionality.

| Layer                   | Output Shape            | Param # | Connected to               |
|-------------------------|-------------------------|---------|----------------------------|
| input_1 (InputLayer)    | (None, 32, 32, 1)       | 0       |                            |
| conv2d                  | (None, 16, 16, 64)      | 3200    | input_1[0][0]              |
| max_pooling2d           | (None, 8, 8, 64)        | 0       | conv2d[0][0]               |
| max_pooling2d_1         | (None, 8, 8, 64)        | 0       | max_pooling2d[0][0]        |
| conv2d_1                | (None, 8, 8, 64)        | 4160    | max_pooling2d[0][0]        |
| conv2d_2                | (None, 8, 8, 128)       | 73856   | max_pooling2d[0][0]        |
| conv2d_3                | (None, 8, 8, 32)        | 51232   | max_pooling2d[0][0]        |
| conv2d_4                | (None, 8, 8, 32)        | 2080    | max_pooling2d_1[0][0]      |
| concatenate             | (None, 8, 8, 256)       | 0       | conv2d_1[0][0],            |
|                         |                         |         | conv2d_2[0][0],            |
|                         |                         |         | conv2d_3[0][0],            |
|                         |                         |         | conv2d_4[0][0]             |
| max_pooling2d_2         | (None, 8, 8, 256)       | 0       | concatenate[0][0]          |
| conv2d_5                | (None, 8, 8, 128)       | 32896   | concatenate[0][0]          |
| conv2d_6                | (None, 8, 8, 192)       | 442560  | concatenate[0][0]          |
| conv2d_7                | (None, 8, 8, 96)        | 614496  | concatenate[0][0]          |
| conv2d_8                | (None, 8, 8, 64)        | 16448   | max_pooling2d_2[0][0]      |
| concatenate_1           | (None, 8, 8, 480)       | 0       | conv2d_5[0][0],            |
|                         |                         |         | conv2d_6[0][0],            |
|                         |                         |         | conv2d_7[0][0],            |
|                         |                         |         | conv2d_8[0][0]             |
| max_pooling2d_3         | (None, 4, 4, 480)       | 0       | concatenate_1[0][0]        |
| dropout                 | (None, 4, 4, 480)       | 0       | max_pooling2d_3[0][0]      |
| global_average_pooling2d| (None, 480)             | 0       | dropout[0][0]              |
| dropout_1               | (None, 480)             | 0       | global_average_pooling2d[0]|
| flatten                 | (None, 480)             | 0       | dropout_1[0][0]            |
| dense                   | (None, 10)              | 4810    | flatten[0][0]              |

Total params: 1245738 (4.75 MB)<br>
Trainable params: 1245738 (4.75 MB)<br>
Non-trainable params: 0 (0.00 Byte)<br>

### Results
The results aren't satisfactory but the objective is to work with this architecture. We test the variables (1 with norm grayscale, 1 with norm grayscale and random noise, and the last nor grayscale and smoothed) all variables were trained with 200 epochs and a batch = 1000. below there is the result on train datasets of values just normalize and transform in grayscale.
![colorful_picture](https://github.com/lucasfreire01/Recognition_iamges_labels/blob/main/reports/performance/performance(1).png)<br>
![colorful_picture](https://github.com/lucasfreire01/Recognition_iamges_labels/blob/main/reports/performance/performance(2).png)<br>

Now the variables with Random Noise the result shown below just in the train datasets.<br>
![colorful_picture](https://github.com/lucasfreire01/Recognition_iamges_labels/blob/main/reports/performance/performance(3).png)<br>
![colorful_picture](https://github.com/lucasfreire01/Recognition_iamges_labels/blob/main/reports/performance/performance(4).png)<br>

Finally, we have the variable Noise smoothed and that is the result in the train dataset
![colorful_picture](https://github.com/lucasfreire01/Recognition_iamges_labels/blob/main/reports/performance/performance(5).png)<br>
![colorful_picture](https://github.com/lucasfreire01/Recognition_iamges_labels/blob/main/reports/performance/performance(6).png)<br>

How can you see in the graphs we have a line more clean and linear now we going to see if this aspect is present in the test datasets. In the variable with normalize and transform in the grayscale have the results: Score = 70,12%
![colorful_picture](https://github.com/lucasfreire01/Recognition_iamges_labels/blob/main/reports/performance/performance(7).png)<br>
![colorful_picture](https://github.com/lucasfreire01/Recognition_iamges_labels/blob/main/reports/performance/performance(8).png)<br>

Variable with normalize, transform in grayscale, and noise random have this score in the test dataset: Score: 65,13%
![colorful_picture](https://github.com/lucasfreire01/Recognition_iamges_labels/blob/main/reports/performance/performance(9).png)<br>
![colorful_picture](https://github.com/lucasfreire01/Recognition_iamges_labels/blob/main/reports/performance/performance(10).png)<br>

This variable has normalized, transformed in grayscale and noise random(smoothed) this is the score: Score: 38,08%
![colorful_picture](https://github.com/lucasfreire01/Recognition_iamges_labels/blob/main/reports/performance/performance(11).png)<br>
![colorful_picture](https://github.com/lucasfreire01/Recognition_iamges_labels/blob/main/reports/performance/performance(12).png)<br>
![colorful_picture](https://github.com/lucasfreire01/Recognition_iamges_labels/blob/main/reports/performance/performance(13).png)<br>

Now we have the model and prediction for each variable so we have a relation between the true value and predicted value:<br>
**variable with normalize and transform in grayscale**<br>
![colorful_picture](https://github.com/lucasfreire01/Recognition_iamges_labels/blob/main/reports/figures/figure(6).png)<br>

**Variable with normalize, transform in grayscale, and noise random.**<br>
![colorful_picture](https://github.com/lucasfreire01/Recognition_iamges_labels/blob/main/reports/figures/figure(6).png)<br>

**Variable with normalize, transform in grayscale, noise random, and smoothed.**<br>
![colorful_picture](https://github.com/lucasfreire01/Recognition_iamges_labels/blob/main/reports/figures/figure(6).png)<br>

## Conclusion
I Liked so much to work with the GoogleNet architecture result so that I wasn't satisfied but the objective is to work and study this architecture, there are some ways to improve the score maybe replacing the architecture, using more pre-processing tools, doing a reverse way we did and reduce the random I think a good way is to use machine learning models to do it and finally to put more neurons in each layer in the neural networking model. If you arrived here thank you so much for seeing my project.
