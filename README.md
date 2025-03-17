# CS24M020 da6401_assignment 1
Report for the project : https://wandb.ai/karekargrishma1234-iit-madras-/cs24m020_dla1_Q4/reports/CS24M020-DA6401-Assignment-1-Report--VmlldzoxMTgyOTc5OA?accessToken=wmdsnvpqdmp4spzk8fqdvx4qcxhpg7mbp7qarz5prfo46ngj34c2rc4jfmacuhwr

# Explanation of the project files:
1) Q2-3 is the jupyter notebook which contains all the necessary functions for training the Neural Network model used in questions 1-10. It contains the code implementations of question 2 and 3 . It contains the optimisers as mentioned like sgd, momentum based GD, nesterov accelerated GD, RMSProp, Adam as asked in question 3. This code only tests on fashion_mnist data.

2) train.py is the python script to be called as that accepts the following command line arguments with the specified values -

# Steps to follow:
Install wandb by command pip install wandb to your system
Run train.py file with the appropriate arguments given below
Your run will be logged to wandb to my project "karekargrishma1234-iit-madras-" and entity="cs24m020_dlA1_Q4_1"
You can view the logs that is the model training accuracy, validation accuracy, testing accuracy and respective losses

# Explanation of implementation of MLFFNN in brief 

This project implements a flexible Feedforward Neural Network (FFNN) from scratch using only NumPy. It is designed for classifying images from the Fashion-MNIST dataset and allows customization of architecture, activation functions, optimization algorithms, and regularization techniques.

Features:

-Fully connected neural network with a customizable number of hidden layers and neurons.

-Activation functions: ReLU, Sigmoid, Tanh (selectable for hidden layers).

-Softmax output layer for multi-class classification.

-Cross-entropy loss function for training.

-Backpropagation algorithm implemented from scratch.

-Multiple optimization algorithms supported:Stochastic Gradient Descent (SGD),Momentum-based Gradient Descent,Nesterov Accelerated Gradient Descent,RMSprop,Adam

-Weight initialization options: Random or Xavier initialization.

-L2 Regularization (Weight Decay) to prevent overfitting.

-Hyperparameter tuning using Weights & Biases (wandb.sweep).

-Mini-batch training support.

-Train-validation-test split:

-Uses 10% of training data as validation.

-Evaluates model performance using accuracy and loss.

Logging meaningful experiment names in WandB.

-Installation

Ensure you have Python and NumPy installed. You also need Weights & Biases (wandb) for hyperparameter tuning.

pip install numpy wandb

-Usage

1. Load the Fashion-MNIST Dataset

Ensure that you have the Fashion-MNIST dataset loaded properly. The network expects images as input and labels as target outputs.

2. Configure Network Parameters

Define the network architecture by specifying:

-Number of hidden layers

-Number of neurons per layer

-Activation functions

-Learning rate and optimizer

-Batch size and regularization parameters

3. Train the Model

Run the training loop, which includes forward propagation, loss computation, backpropagation, and weight updates using the chosen optimization algorithm.

4. Evaluate the Model

After training, test the model on unseen data to measure its generalization performance.

# Hyperparameter Tuning

The implementation supports hyperparameter tuning using wandb.sweep. Define a sweep configuration specifying ranges for:

-Learning rate

-Number of layers and neurons

-Activation functions

-Optimizers

-Regularization strength

Run the sweep to find the optimal hyperparameters efficiently.

# Results

The model logs training loss, validation accuracy, and test performance to Weights & Biases, enabling easy visualization of the learning progress.

# Customization

Modify the number of layers and neurons in initialize_weights().

Change activation functions in the forward pass.

Extend the code with additional optimization techniques.

# Conclusion

This FFNN provides a flexible foundation for deep learning experiments while ensuring full control over training dynamics. Future enhancements could include:

Dropout for regularization

Batch normalization

Support for additional datasets
