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

![colorful_picture](https://github.com/lucasfreire01/Recognition_iamges_labels/blob/main/download1.png)<br>
Bellow there is an example of the images and your labels.<br>
![label_picture](https://github.com/lucasfreire01/Recognition_iamges_labels/blob/main/download2.png)<br>

## Pre_Processing
### Type and grayscale
The first thing did was transform the type of values from int to float32 because processing is more computationally cheap, next used coding in the y_train and y_test and transformed in the grayscale because images are 3 buses (R, G, B) for example (187, 222, 87) when we get a gray image in RGB will be for example(187, 187, 187) we have gray in the RGB base when all values are the same the color is gray this enables us to use 1 buses because the number is the same that is the 3 buses(R G B), bellow is an example in the gray. After we norm these values based on the white number in RGB (255). Here there is an example of grayscale.<br>
![colorful_picture](https://github.com/lucasfreire01/Recognition_iamges_labels/blob/main/download3.png)<br>
![gray_picture](https://github.com/lucasfreire01/Recognition_iamges_labels/blob/main/download4.png)<br>
### Noise_Random
We dds random noise to a normalized grayscale training dataset (x_train_norm_gray). The noise is generated using a normal distribution with a standard deviation of 0.1. The purpose is to introduce variability into the data, aiding the model in learning to be more robust by adapting to different patterns and reducing overfitting. This technique, known as data augmentation, enhances the model's generalization capabilities without requiring additional labeled data.<br>
![colorful_picture](https://github.com/lucasfreire01/Recognition_iamges_labels/blob/main/download5.png)<br>
![colorful_picture](https://github.com/lucasfreire01/Recognition_iamges_labels/blob/main/download6.png)<br>

