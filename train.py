import argparse

import numpy as np
import pandas as pd
import wandb
import matplotlib.pyplot as plt
from keras.datasets import fashion_mnist
from keras.datasets import mnist
import seaborn as sns

#wandb.login()

class Start:
  def __init__(self,dataset='fashion_mnist'):
    if dataset=='fashion_mnist':
      training_data,testing_data = fashion_mnist.load_data()
    else:
      training_data,testing_data = mnist.load_data()
    self.train_x,self.train_y=training_data
    self.test_x,self.test_y=testing_data

    self.val_x = None
    self.val_y = None


  def data(self):
    return self.train_x,self.train_y,self.test_x,self.test_y

  def split_data(self):
    split_index = int(0.9 * self.train_x.shape[0])
    self.x_train_split, self.val_x =self. train_x[:split_index], self.train_x[split_index:]
    self.y_train_split, self.val_y =self. train_y[:split_index], self.train_y[split_index:]
    return self.x_train_split,self.val_x,self.y_train_split,self.val_y

  def modified_data(self):
    train_x=self.x_train_split.reshape(-1,28*28)/255.0
    test_x=self.test_x.reshape(-1,28*28)/255.0
    val_x=self.val_x.reshape(-1,28*28)/255.0
    train_y=np.eye(10)[self.y_train_split]
    test_y=np.eye(10)[self.test_y]
    val_y=np.eye(10)[self.val_y]
    return train_x,test_x,val_x,train_y,test_y,val_y

  def normalize_data(self):
    train_x=self.x_train_split.reshape(-1,28*28)/255.0
    test_x=self.test_x.reshape(-1,28*28)/255.0
    val_x=self.val_x.reshape(-1,28*28)/255.0
    return train_x,test_x,val_x

#Activation Functions
class Activation_Functions:
    def __init__(self, act):
        self.af = act.lower()

    #For Hidden Layers
    def activation_function(self, z):
        if self.af == 'relu':
            return np.maximum(0, z)
        elif self.af == 'tanh':
            z=np.clip(z,-500,500)
            return np.tanh(z)
        elif self.af == 'sigmoid':
            z=np.clip(z,-500,500)
            return 1 / (1 + np.exp(-z))
        elif self.af == 'identity':
            return z

    def activation_derivative(self, a):
        if self.af == 'relu':
            return (a > 0).astype(float)
        elif self.af == 'tanh':
            return 1 - a ** 2
        elif self.af == 'sigmoid':
            return a * (1 - a)
        elif self.af == 'identity':
            return np.ones_like(a)

    def ad2(self, z):
        if self.af == 'relu':
            return (z > 0).astype(int)
        elif self.af == 'tanh':
            ac = self.activation_function(z)
            return 1 - ac ** 2
        elif self.af == 'sigmoid':
            ac = self.activation_function(z)
            return ac * (1 - ac)
        elif self.af == 'identity':
            return np.ones_like(z)
    #For Output Layer
    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

#Intialization methods : To intialize weights
class Initializer_Methods:
  def __init__(self,m):
    self.mthd=m.lower()

  def initialize_weights(self, fin, fout, act):
       if self.mthd == 'xavier':
        if(act=='relu'):
          return np.random.randn(fin, fout) * np.sqrt(2 / fin)
        return np.random.randn(fin, fout) * np.sqrt(1 / fin)
       elif self.mthd == 'random':
        return np.random.randn(fin, fout) * 0.01

#Error functions and Accuracy calculation
class Eval_Metrics:
  def __init__(self):
      pass

  def cross_entropy_loss(self, y_true, y_pred):
      return -np.sum(y_true * np.log(y_pred + 1e-9)) / y_true.shape[0]

  def squared_error(self,y_true,y_pred):
      return np.mean(np.sum(np.square(y_true - y_pred), axis=1))

  def accuracy(self, y_true, y_pred):
      return np.mean(np.argmax(y_true, axis=1) == np.argmax(y_pred, axis=1))


class FeedForwardNN:
    def __init__(self, input_size, hidden_layers, output_size, learning_rate=0.001, activation='sigmoid', weight_init='random', weight_decay=0.0 , error_type='cross'):
        
        self.layers = [input_size] + hidden_layers + [output_size]
        self.activation = activation.lower()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.error=error_type
        self.weight_init=weight_init
        self.t = 0
        self.activations = []
        self.d_weights = []
        self.d_biases = []


        ac=Activation_Functions(self.activation)
        self.af=ac.activation_function
        self.ad1=ac.activation_derivative
        self.ad2=ac.ad2
        self.softmax=ac.softmax

        w_init=Initializer_Methods(self.weight_init)
        self.initialize_weights = w_init.initialize_weights

        eval=Eval_Metrics()
        self.cross_entropy_loss=eval.cross_entropy_loss
        self.squared_error=eval.squared_error
        self.accuracy=eval.accuracy


        # Initialize weights and biases
        self.weights = [self.initialize_weights(self.layers[i], self.layers[i + 1], weight_init) for i in range(len(self.layers) - 1)]
        self.biases = [np.zeros((1, self.layers[i + 1])) for i in range(len(self.layers) - 1)]

        # For the optimizers, initialize the necessary variables
        self.momentums = [np.zeros_like(w) for w in self.weights]
        self.biases_momentums = [np.zeros_like(b) for b in self.biases]
        self.velocity = [np.zeros_like(w) for w in self.weights]
        self.squared_gradients = [np.zeros_like(w) for w in self.weights]
        self.squared_biases = [np.zeros_like(w) for w in self.biases]
        self.m = [np.zeros_like(w) for w in self.weights]
        self.v = [np.zeros_like(w) for w in self.weights]
        self.biases_m = [np.zeros_like(w) for w in self.biases]
        self.biases_v = [np.zeros_like(w) for w in self.biases]


    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        a = x
        self.activations = [a]
        for w, b in zip(self.weights[:-1], self.biases[:-1]):
            a = self.af(np.dot(a, w) + b)
            self.activations.append(a)
        output = np.dot(a, self.weights[-1]) + self.biases[-1]
        self.activations.append(output)
        return self.softmax(output)

    def backward(self, x, y):
        m = x.shape[0]
        if(self.error=='cross_entropy'):
          d_output = self.activations[-1] - y
        elif(self.error=='mean_squared_error'):
          d_output =( (self.activations[-1] - y)*2 ) / m

        d_w = np.dot(self.activations[-2].T, d_output) / m + self.weight_decay * self.weights[-1]
        d_b = np.sum(d_output, axis=0, keepdims=True) / m

        self.d_weights = [d_w]
        self.d_biases = [d_b]

        for i in range(len(self.layers) - 3, -1, -1):
            #d_activation = np.dot(d_output, self.weights[i + 1].T) * self.activation_derivative(np.dot(self.activations[i], self.weights[i]) + self.biases[i])
            d_activation = np.dot(d_output, self.weights[i + 1].T) * self.ad2(self.activations[i + 1])
            d_output = d_activation
            d_w = np.dot(self.activations[i].T, d_output) / m + self.weight_decay * self.weights[i]
            d_b = np.sum(d_output, axis=0, keepdims=True) / m
            self.d_weights.insert(0, d_w)
            self.d_biases.insert(0, d_b)

    def update_weights(self, optimizer="sgd"):
        """Update weights and biases using the specified optimization algorithm."""
        if optimizer == "sgd":
            self.sgd_optimizer()
        elif optimizer == "momentum":
            self.momentum_optimizer()
        elif optimizer == "nesterov":
            self.nesterov_optimizer()
        elif optimizer == "rmsprop":
            self.rmsprop_optimizer()
        elif optimizer == "adam":
            self.adam_optimizer()
        elif optimizer == "nadam":
            self.nadam_optimizer()

    #Optimizers
    def sgd_optimizer(self):
        """SGD update rule."""
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * (self.d_weights[i]+self.weight_decay*self.weights[i])
            self.biases[i] -= self.learning_rate * self.d_biases[i]

    def momentum_optimizer(self, beta=0.9):
      """Momentum-based gradient descent update."""
      for i in range(len(self.weights)):
          self.momentums[i] = beta * self.momentums[i] + self.learning_rate * (self.d_weights[i]+self.weight_decay*self.weights[i])
          self.weights[i] -= self.learning_rate * self.momentums[i]
          self.biases_momentums[i] = beta * self.biases_momentums[i] + self.learning_rate * self.d_biases[i]
          self.biases[i] -= self.learning_rate * self.biases_momentums[i]


    def nesterov_optimizer(self, beta=0.9):
      """Nesterov Accelerated Gradient Descent (NAG)."""
      for i in range(len(self.weights)):
          lookahead_weights = self.weights[i] - beta * self.momentums[i]
          grad_w = self.d_weights[i]
          self.momentums[i] = beta * self.momentums[i] + self.learning_rate * (grad_w + self.weight_decay*lookahead_weights)
          self.weights[i] = lookahead_weights - self.learning_rate * self.momentums[i]

          lookahead_biases = self.biases[i] - beta * self.biases_momentums[i]
          grad_b = self.d_biases[i]
          self.biases_momentums[i] = beta * self.biases_momentums[i] + self.learning_rate * grad_b
          self.biases[i] = lookahead_biases - self.learning_rate * self.biases_momentums[i]


    def rmsprop_optimizer(self, beta=0.9, epsilon=1e-6):
      """RMSprop optimizer (fixed bias update issue)."""
      for i in range(len(self.weights)):
          self.squared_gradients[i] = beta * self.squared_gradients[i] + (1 - beta) * (self.d_weights[i] ** 2)
          self.squared_biases[i] = beta * self.squared_biases[i] + (1 - beta) * (self.d_biases[i] ** 2)
          self.weights[i] -= self.learning_rate * ((self.d_weights[i]+ (self.weight_decay * self.weights[i])) / (np.sqrt(self.squared_gradients[i]) + epsilon))
          self.biases[i] -= self.learning_rate * (self.d_biases[i] / (np.sqrt(self.squared_biases[i]) + epsilon))

    def adam_optimizer(self, beta1=0.9, beta2=0.999, epsilon=1e-6):
      """Adam optimizer."""
      self.t += 1
      for i in range(len(self.weights)):
          self.m[i] = beta1 * self.m[i] + (1 - beta1) * self.d_weights[i]
          self.v[i] = beta2 * self.v[i] + (1 - beta2) * (self.d_weights[i] ** 2)
          m_hat = self.m[i] / (1 - beta1 ** self.t)
          v_hat = self.v[i] / (1 - beta2 ** self.t)
          self.weights[i] -= self.learning_rate * (( m_hat /( (np.sqrt(v_hat) + epsilon)) + self.weight_decay * self.weights[i]) )

          self.biases_m[i] = beta1 * self.biases_m[i] + (1 - beta1) * self.d_biases[i]
          self.biases_v[i] = beta2 * self.biases_v[i] + (1 - beta2) * (self.d_biases[i] ** 2)
          biases_m_hat = self.biases_m[i] / (1 - beta1 ** self.t)
          biases_v_hat = self.biases_v[i] / (1 - beta2 ** self.t)
          self.biases[i] -= self.learning_rate * (( biases_m_hat / (np.sqrt(biases_v_hat) + epsilon))  )


    def nadam_optimizer(self, beta1=0.9, beta2=0.999, epsilon=1e-6):
      """Nadam optimizer (combining Nesterov and Adam)."""
      self.t += 1  # Time step for bias correction

      for i in range(len(self.weights)):
          self.m[i] = beta1 * self.m[i] + (1 - beta1) * self.d_weights[i]
          self.v[i] = beta2 * self.v[i] + (1 - beta2) * (self.d_weights[i] ** 2)
          m_hat = self.m[i] / (1 - (beta1 ** self.t))
          v_hat = self.v[i] / (1 - (beta2 ** self.t))
          nesterov_m = beta1 * m_hat + (1 - beta1) * self.d_weights[i]
          self.weights[i] -= self.learning_rate * (( nesterov_m / (np.sqrt(v_hat) + epsilon))+ self.weight_decay * self.weights[i] )

          self.biases_m[i] = beta1 * self.biases_m[i] + (1 - beta1) * self.d_biases[i]
          self.biases_v[i] = beta2 * self.biases_v[i] + (1 - beta2) * (self.d_biases[i] ** 2)
          biases_m_hat = self.biases_m[i] / (1 - (beta1 ** self.t))
          biases_v_hat = self.biases_v[i] / (1 - (beta2 ** self.t))
          nesterov_bias_m = beta1 * biases_m_hat + (1 - beta1) * self.d_biases[i]
          self.biases[i] -= self.learning_rate *( nesterov_bias_m / (np.sqrt(biases_v_hat) + epsilon) )


    def train(self, x_train, y_train, epochs, batch_size, optimizer="sgd", validation_data=None):
      num_samples = x_train.shape[0]

      # Moving average for validation loss/accuracy
      val_loss_history = []
      val_acc_history = []

      for epoch in range(epochs):
          total_loss = 0
          total_acc = 0
          permutation = np.random.permutation(num_samples)
          x_train = x_train[permutation]
          y_train = y_train[permutation]

          # Mini-batch training loop
          for i in range(0, num_samples, batch_size):
              x_batch = x_train[i:i+batch_size]
              y_batch = y_train[i:i+batch_size]
              output = self.forward(x_batch)

              # Compute loss
              if self.error == 'cross_entropy':
                  loss = self.cross_entropy_loss(y_batch, output)
              elif self.error == 'mean_squared_error':
                  loss = self.squared_error(y_batch, output)

              acc = self.accuracy(y_batch, output)
              total_loss += loss
              total_acc += acc

              # Backpropagation and weight update
              self.backward(x_batch, y_batch)
              self.update_weights(optimizer)

          # Compute epoch-level training loss & accuracy
          avg_train_loss = total_loss / (num_samples // batch_size)
          avg_train_acc = (total_acc / (num_samples // batch_size)) * 100

          # Validation computation
          val_loss_no, val_acc_no = None, None
          if validation_data:
              x_val, y_val = validation_data
              val_output = self.forward(x_val)

              # Compute validation loss
              if self.error == 'cross_entropy':
                  val_loss = self.cross_entropy_loss(y_val, val_output)
              elif self.error == 'mean_squared_error':
                  val_loss = self.squared_error(y_val, val_output)

              val_acc = self.accuracy(y_val, val_output) * 100

              # Moving average for stability
              val_loss_history.append(val_loss)
              val_acc_history.append(val_acc)

              if len(val_loss_history) > 5:  # Maintain a window of last 5 epochs
                  val_loss_history.pop(0)
                  val_acc_history.pop(0)

              val_loss_no = sum(val_loss_history) / len(val_loss_history)
              val_acc_no = sum(val_acc_history) / len(val_acc_history)

          # Print results
          print(f"Epoch: {epoch+1}/{epochs}, Loss: {avg_train_loss:.4f}, Accuracy: {avg_train_acc:.2f}%, "
                f"Val Loss: {val_loss_no:.4f}, Val Accuracy: {val_acc_no:.2f}%")

          # Logging to WandB
          wandb.log({"epoch": epoch+1, "loss": avg_train_loss, "accuracy": avg_train_acc})
          if validation_data:
              wandb.log({"val_loss": val_loss_no, "val_acc": val_acc_no})

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb_entity", "-we",help = "Wandb Entity used to track experiments in the Weights & Biases dashboard.", default="cs24m020")
    parser.add_argument("--wandb_project", "-wp",help="Project name used to track experiments in Weights & Biases dashboard", default="Trial")
    parser.add_argument("--dataset", "-d", help = "dataset", choices=["mnist","fashion_mnist"])
    parser.add_argument("--epochs","-e", help= "Number of epochs to train neural network", type= int, default=10)
    parser.add_argument("--batch_size","-b",help="Batch size used to train neural network", type =int, default=32)
    parser.add_argument("--optimizer","-o",help="batch size is used to train neural network", default= "adam", choices=["sgd","momentum","nag","rmsprop","adam","nadam"])
    parser.add_argument("--loss","-l", default= "cross_entropy", choices=["mean_squared_error", "cross_entropy"])
    parser.add_argument("--learning_rate","-lr", default=0.01, type=float)
    parser.add_argument("--momentum","-m", default=0.5,type=float)
    parser.add_argument("--beta","-beta", default=0.5, type=float)
    parser.add_argument("--beta1","-beta1", default=0.9,type=float)
    parser.add_argument("--beta2","-beta2", default=0.999,type=float)
    parser.add_argument("--epsilon","-eps",type=float, default = 0.0000001)
    parser.add_argument("--weight_decay","-w_d", default=0.005,type=float)
    parser.add_argument("-w_i","--weight_init", default="xavier",choices=["random","xavier"])
    parser.add_argument("--num_layers","-nhl",type=int, default=4)
    parser.add_argument("--hidden_size","-sz",type=int, default=64)
    parser.add_argument("-a","--activation",choices=["identity","sigmoid","tanh","relu"], default="ReLU")

    args = parser.parse_args()
    print(args.epochs)
    wandb.login()
    wandb.init(project=args.wandb_project,entity=args.wandb_entity)

    x=Start(dataset=args.dataset)
    trainx,testx,trainy,testy=x.data()
    trainx,valx,trainy,valy=x.split_data()
    x_train,x_test,x_val,y_train,y_test,y_val = x.modified_data()

    model = FeedForwardNN(input_size=28*28, hidden_layers=[args.hidden_size]*args.num_layers, output_size=10, learning_rate=args.learning_rate, 
                          activation=args.activation, weight_init=args.weight_init, weight_decay=args.weight_decay,error_type=args.loss)
    
    model.train(x_train, y_train, args.epochs, args.batch_size, optimizer=args.optimizer, validation_data=(x_val,y_val))
    
    wandb.finish()

