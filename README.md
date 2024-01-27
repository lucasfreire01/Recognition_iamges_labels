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
All datasets have this matrix each set of cords represents an image and each cord with 3 values(RGB) in the training dataset we have 50000 values, the X_train we get the shape(50000, 32, 32, 3) and in the test datasets we get 10000 values the x_test have the shape (10000, 32, 32, 3) the rest follow this rule.
