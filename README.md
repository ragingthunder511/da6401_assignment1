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

# Introduction
The project implements a feedforward neural network from scratch using NumPy. It supports:
- Datasets: Fashion-MNIST and MNIST.
- Activation Functions: ReLU, Sigmoid, Tanh, and Identity.
- Optimizers: SGD, Momentum, Nesterov Accelerated Gradient (NAG), RMSProp, Adam, and Nadam.
- Weight Initialization: Random and Xavier.
- Loss Functions: Cross-Entropy and Mean Squared Error.

The model is trained and evaluated using a train-validation-test split, and performance metrics are logged using Weights & Biases (wandb).

# Requirements
To run this code, you need the following Python libraries:
- numpy
- wandb
- keras (for dataset loading)
- argparse
- matplotlib
- seaborn

You can install the required libraries using:
pip install numpy wandb keras matplotlib seaborn

# Installation
1) Clone the repository:
git clone https://github.com/ragingthunder511/da6401_assignment1.git
cd da6401_assignment1
2) Install the required libraries

# Usage
To train the model, run the train.py script with the desired arguments. For example:
python train.py --dataset fashion_mnist --epochs 20 --batch_size 64 --optimizer adam --learning_rate 0.001 --activation relu --weight_init xavier

# Parameters 

| Name                      | Default Value | Description                                                                 |
|---------------------------|---------------|-----------------------------------------------------------------------------|
| `-wp`, `--wandb_project`  | `myprojectname` | Project name used to track experiments in Weights & Biases dashboard.       |
| `-we`, `--wandb_entity`   | `myname`      | WandB Entity used to track experiments in the Weights & Biases dashboard.   |
| `-d`, `--dataset`         | `fashion_mnist` | Dataset to use. Choices: `["mnist", "fashion_mnist"]`.                      |
| `-e`, `--epochs`          | `1`           | Number of epochs to train neural network.                                   |
| `-b`, `--batch_size`      | `4`           | Batch size used to train neural network.                                    |
| `-l`, `--loss`            | `cross_entropy` | Loss function. Choices: `["mean_squared_error", "cross_entropy"]`.          |
| `-o`, `--optimizer`       | `sgd`         | Optimization algorithm. Choices: `["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"]`. |
| `-lr`, `--learning_rate`  | `0.1`         | Learning rate used to optimize model parameters.                            |
| `-m`, `--momentum`        | `0.5`         | Momentum used by momentum and NAG optimizers.                               |
| `-beta`, `--beta`         | `0.5`         | Beta used by RMSProp optimizer.                                             |
| `-beta1`, `--beta1`       | `0.5`         | Beta1 used by Adam and Nadam optimizers.                                    |
| `-beta2`, `--beta2`       | `0.5`         | Beta2 used by Adam and Nadam optimizers.                                    |
| `-eps`, `--epsilon`       | `0.000001`    | Epsilon used by optimizers.                                                 |
| `-w_d`, `--weight_decay`  | `.0`          | Weight decay used by optimizers.                                            |
| `-w_i`, `--weight_init`   | `random`      | Weight initialization method. Choices: `["random", "xavier"]`.              |
| `-nhl`, `--num_layers`    | `1`           | Number of hidden layers used in feedforward neural network.                 |
| `-sz`, `--hidden_size`    | `4`           | Number of hidden neurons in a feedforward layer.                            |
| `-a`, `--activation`      | `sigmoid`     | Activation function. Choices: `["identity", "sigmoid", "tanh", "relu"]`.    |

# Code Structure
The repository is organized as follows:
- train.py: Main script for training the model.
Classes:
- Start: Handles dataset loading, splitting, and preprocessing.
- Activation_Functions: Implements activation functions and their derivatives.
- Initializer_Methods: Implements weight initialization methods.
- Eval_Metrics: Implements loss functions and accuracy calculation.
- FeedForwardNN: Implements the neural network, including forward/backward propagation and optimization.

# Results
The training and validation loss/accuracy are logged using Weights & Biases (wandb). You can visualize the results on the wandb dashboard.
To view the results:
- Create a wandb account at wandb.ai.
- Log in using the wandb login command.
- Run the script with your desired parameters.
- View the results on your wandb dashboard.

# Contributing
Contributions are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.



