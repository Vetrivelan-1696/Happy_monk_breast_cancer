# Happy_monk_breast_cancer
Classifiation of Breast cancer using ANN

# Prerequistes :
The first thing we'll do inside our Jupyter Notebook is import various open-source libraries that we'll use throughout our Python script, including NumPy, matplotlib, pandas, and (most importantly) TensorFlow. Run the following import statements to start your script:

# Importing the Data Set :
The next thing we'll do is import our data set into the Python script we're working on. More specifically, we will store the data set in a pandas DataFrame using the read_csv method.

# Data Pre Processing :
Both the x_data and y_data variables are NumPy arrays that contain the x-values (also called our features) and the y-data (also called our labels) that we'll use to train our artificial neural network later.

Before we can train our data, we must first make some modifications to the categorical data within our data set.

# Splitting The Data Set Into Training Data and Test Data:

Machine learning practitioners almost always use scikit-learn's built-in train_test_split function to split their data set into training data and test data.
The train_test_split function returns a Python list of length 4 with the following items:

The x training data
The x test data
The y training data
The y test data
train_test_split is typically combined with list unpacking to easily create 4 new variables that store each of the list's items. As an example, here's how we'll create our training and test data using a test_size parameter of 0.3 (which simply means that the test data will be 30% of the observations of the original data set).

# Feature Scaling :
The next thing we need to do is feature scaling, which is the process of modifying our independent variables so that they are all roughly the same size.

Feature scaling is absolutely critical for deep learning. While many statistical methods benefit from feature scaling, it is actually required for deep learning.

We'll apply feature scaling to every feature of our data set because of this. To start, import the StandardScaler class from scikit-learn and create an instance of this class.

# Building the Artificial Neural Network :
We will follow four broad steps to build our artificial neural network:

Initializing the Artificial Neural Network
Adding The Input Layer & The First Hidden Layer
Adding The Second Hidden Layer
Adding The Output Layer
Let's go through each of these steps one-by-one.

# Initializing the Neural Network :
Let's break down what's happening here:

We're creating an instance of the Sequential class
The Sequential class lives within the models module of the keras library
Since TensorFlow 2.0, Keras is now a part of TensorFlow, so the Keras package must be called from the tf variable we created earlier in our Python script
All of this code serves to create a "blank" artificial neural network.

# Adding the input and hidden layer :
Now it's time to add our input layer and our first hidden layer.

Let's start by discussing the input layer. No action is required here. Neural network input layers do not need to actually be created by the engineer building the network.

Why is this?

Well, if you think back to our discussion of input layers, remember that they are entirely decided by the data set that the model is trained on. We do not need to specify our input layer because of this.

With that in mind, we can move on to adding the hidden layer.

Layers can be added to a TensorFlow neural network using the add method. To start, call the add method on our ann variable without passing in any parameters:

# Adding the Second hidden layer :
Adding a second hidden layer follows the exact same process as the original hidden layer. Said differently, the add method does not need to be used specifically for the first hidden layer of a neural network!

Let's add another hidden layer with 6 units that has a ReLU activation function:

# Adding the output layer :
Like the hidden layers that we added earlier in this tutorial, we can add our output layer to the neural network with the add function. However, we'll need to make several modifications to the statement we used previously.

# Training The Artificial Neural Network
Our artificial neural network has been built. Now it's time to train the model using the training data we created earlier in this tutorial. This process is divided into two steps:

Compiling the neural network
Training the neural network using our training data
Compiling The Neural Network
In deep learning, compilation is a step that transforms the simple sequence of layers that we previously defined into a highly efficient series of matrix transformations. You can interpret compilation as a precompute step that makes it possible for the computer to train the model.

TensorFlow allows us to compile neural networks using its compile function, which requires three parameters:

The optimizer
The cost function
The metrics parameter
Let's first create a blank compile method call that includes these three metrics (without specifying them yet):

ann.compile(optimizer = , loss = , metrics = )
Let's start by selecting our optimizer. We will be using the Adam optimizer, which is a highly performant stochastic gradient descent algorithm descent specifically for training deep neural networks.

# Making Predictions With Our Artificial Neural Network
Now that our artificial neural network has been trained, we can use it to make predictions using specified data points. We do this using the predict method.

Before we start, you should note that anything passed into a predict method called on an artificial neural network built using TensorFlow needs to be a two-dimensional array. 

# Measuring The Performance Of The Artificial Neural Network Using The Test Data
The last thing we'll do in this tutorial is measure the performance of our artificial neural network on our test data.

To start, let's generate an array of boolean values that predicts whether every customer in our test data will churn or not. We will assign this array to a variable called predictions.

You might also want to calculate the accuracy of our model, which is the percent of predictions that were correct. scikit-learn has a built-in function called accuracy_score to 

# Evaluating and finding out best activation function :
FInally evaluating all the activation function by its f1 score , accuracy score and found out the best optimized activation function for respective epochs for this specific output.

Thank you ....!!
