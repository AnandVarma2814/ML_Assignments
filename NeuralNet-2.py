#####################################################################################################################
#   Assignment 2: Neural Network Analysis
#   This is a starter code in Python 3.6 for a neural network.
#   You need to have numpy and pandas installed before running this code.
#   You need to complete all TODO marked sections
#   You are free to modify this code in any way you want, but need to mention it
#       in the README file.
#
#####################################################################################################################


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from tensorflow.keras import optimizers
from keras.layers import Dense
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error,r2_score
import matplotlib.pyplot as plt
# , Activation,Layer,Lambda


class NeuralNet:
    def __init__(self, dataFile, header=True):
        self.raw_input = pd.read_csv(dataFile, header = None)
        self.MSE_train = []
        self.MSE_test = []
        self.r2_train = []
        self.r2_test = []

    # TODO: Write code for pre-processing the dataset, which would include
    # standardization, normalization,
    #   categorical to numerical, etc
    def checkRemoveDuplicates(self, data):
        print("Number of duplicated records in given data: ", data.duplicated().sum())
        data = data.drop_duplicates()
        return data

    def checkReplaceNull(self, data):
        print("Number of null values in given data: ", data.isnull().sum().sum())
        return data

    def encodeY(self, data, column):
        le = preprocessing.LabelEncoder()
        data.iloc[:, column] = le.fit_transform(data.iloc[:, column])
        return data

    def preprocess(self):
        remDup = self.checkRemoveDuplicates(self.raw_input)
        remNull = self.checkReplaceNull(remDup)
        self.processed_data = self.encodeY(remNull, 0)
        return 0

    # TODO: Train and evaluate models for all combinations of parameters
    # specified. We would like to obtain following outputs:
    #   1. Training Accuracy and Error (Loss) for every model
    #   2. Test Accuracy and Error (Loss) for every model
    #   3. History Curve (Plot of Accuracy against training steps) for all
    #       the models in a single plot. The plot should be color coded i.e.
    #       different color for each model

    def splitAttributs(self, data, column):
        X = data.iloc[:, data.columns != column]
        Y = data.iloc[:, data.columns == column]
        return (X, Y)

    def train_model(self, act_fn, l_rate, epoch):
        model1 = Sequential()
        if(self.num_hidden_layers == 2):
            hidden_dim=[16,6,3,1]
        elif(self.num_hidden_layers == 3):
            hidden_dim=[16,6,4,2,1]

        for i in range(1,len(hidden_dim)-1):
            if (i==1):
                model1.add(Dense(hidden_dim[i], input_dim = hidden_dim[0], kernel_initializer = "normal", activation = act_fn))
            else:
                model1.add(Dense(hidden_dim[i], activation = act_fn))
        model1.add(Dense(hidden_dim[-1]))
        model1.compile(loss = "mean_squared_error", optimizer = optimizers.Adam(learning_rate = l_rate), metrics = ["accuracy"])
        
        model1.fit(np.array(self.X_train),np.array(self.y_train),epochs = epoch)
        pred1_train = model1.predict(np.array(self.X_train))
        self.MSE_train.append(np.sqrt(mean_squared_error(self.y_train,pred1_train)))
        self.r2_train.append(r2_score(self.y_train,pred1_train))
        
        pred1_test = model1.predict(np.array(self.X_test))
        self.MSE_test.append(np.sqrt(mean_squared_error(self.y_test,pred1_test)))
        self.r2_test.append(r2_score(self.y_test,pred1_test))

    def train_evaluate(self):
        X, y = self.splitAttributs(self.processed_data, 0)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y)

        # Below are the hyperparameters that you need to use for model evaluation
        activations = ['logistic', 'tanh', 'relu']
        learning_rate = [0.01, 0.1]
        max_iterations = [100, 200] # also known as epochs
        num_hidden_layers = [2, 3]

        self.num_hidden_layers = 2
        self.train_model("sigmoid", 0.01, 100)
        self.train_model("sigmoid", 0.01, 200)
        self.train_model("sigmoid", 0.1, 100)
        self.train_model("sigmoid", 0.1, 200)
        self.train_model("tanh", 0.01, 100)
        self.train_model("tanh", 0.01, 200)
        self.train_model("tanh", 0.1, 100)
        self.train_model("tanh", 0.1, 200)
        self.train_model("relu", 0.01, 100)
        self.train_model("relu", 0.01, 200)
        self.train_model("relu", 0.1, 100)
        self.train_model("relu", 0.1, 200)

        self.num_hidden_layers = 3
        self.train_model("sigmoid", 0.01, 100)
        self.train_model("sigmoid", 0.01, 200)
        self.train_model("sigmoid", 0.1, 100)
        self.train_model("sigmoid", 0.1, 200)
        self.train_model("tanh", 0.01, 100)
        self.train_model("tanh", 0.01, 200)
        self.train_model("tanh", 0.1, 100)
        self.train_model("tanh", 0.1, 200)
        self.train_model("relu", 0.01, 100)
        self.train_model("relu", 0.01, 200)
        self.train_model("relu", 0.1, 100)
        self.train_model("relu", 0.1, 200)

        # Create the neural network and be sure to keep track of the performance metrics

        activations = ['logistic','logistic','logistic','logistic','tanh','tanh','tanh','tanh','relu','relu','relu','relu',\
                        'logistic','logistic','logistic','logistic','tanh','tanh','tanh','tanh','relu','relu','relu','relu',]
        learning_rate = [0.01,0.01,0.1,0.1,0.01,0.01,0.1,0.1,0.01,0.01,0.1,0.1,\
                            0.01,0.01,0.1,0.1,0.01,0.01,0.1,0.1,0.01,0.01,0.1,0.1]
        hidden_layers = [2,2,2,2,2,2,2,2,2,2,2,2,\
                            3,3,3,3,3,3,3,3,3,3,3,3]
        max_iter=[100,200,100,200,100,200,100,200,100,200,100,200,\
                    100,200,100,200,100,200,100,200,100,200,100,200]
        temp = {'activation' : activations, 'learning_rate' : learning_rate, 'Hidden_layers' : hidden_layers, 'max_iter' : max_iter, \
                'MSE_train' : self.MSE_train, 'MSE_test' : self.MSE_test,'r2_train' : self.r2_train, 'r2_test' : self.r2_test}

        table = pd.DataFrame(data = temp)

        print("MSE Train: ", self.MSE_train)
        print("MSE Test: ", self.MSE_test)
        print("R2 Train: ", self.r2_train)
        print("R2 Test: ", self.r2_test)
        
        return table

if __name__ == "__main__":
    neural_network = NeuralNet("letter-recognition.data") # put in path to your file
    neural_network.preprocess()
    table = neural_network.train_evaluate()

    # Plot the model history for each model in a single plot
    # model history is a plot of accuracy vs number of epochs
    # you may want to create a large sized plot to show multiple lines
    # in a same figure.

    plt.plot( 'max_iter', 'MSE_train', data=table.loc[(table['activation']=='logistic')&(table['learning_rate']==0.01)&(table['Hidden_layers']==2)], markersize=12, color='red', linewidth=2,label='logistic_0.01_2')    
    plt.plot( 'max_iter', 'MSE_train', data=table.loc[(table['activation']=='logistic')&(table['learning_rate']==0.01)&(table['Hidden_layers']==3)], marker='', color='olive', linewidth=2,label = 'logistic_0.01_3')
    plt.plot( 'max_iter', 'MSE_train', data=table.loc[(table['activation']=='logistic')&(table['learning_rate']==0.1)&(table['Hidden_layers']==2)], markersize=12, color='blue', linewidth=2,label = 'logistic_0.1_2')
    plt.plot( 'max_iter', 'MSE_train', data=table.loc[(table['activation']=='logistic')&(table['learning_rate']==0.1)&(table['Hidden_layers']==3)], marker='', color='orange',linewidth=2, label = 'logistic_0.1_3')
    plt.xlabel('Iterations')
    plt.ylabel('Mean Squared Error')
    plt.legend(loc = 'upper right')
    plt.show()
    plt.plot( 'max_iter', 'MSE_test', data=table.loc[(table['activation']=='logistic')&(table['learning_rate']==0.01)&(table['Hidden_layers']==2)], markersize=12, color='red', linewidth=2,label='logistic_0.01_2')    
    plt.plot( 'max_iter', 'MSE_test', data=table.loc[(table['activation']=='logistic')&(table['learning_rate']==0.01)&(table['Hidden_layers']==3)], marker='', color='olive', linewidth=2,label = 'logistic_0.01_3')
    plt.plot( 'max_iter', 'MSE_test', data=table.loc[(table['activation']=='logistic')&(table['learning_rate']==0.1)&(table['Hidden_layers']==2)], markersize=12, color='blue', linewidth=2,label = 'logistic_0.1_2')
    plt.plot( 'max_iter', 'MSE_test', data=table.loc[(table['activation']=='logistic')&(table['learning_rate']==0.1)&(table['Hidden_layers']==3)], marker='', color='orange',linewidth=2, label = 'logistic_0.1_3')
    plt.xlabel('Iterations')
    plt.ylabel('Mean Squared Error')
    plt.legend(loc = 'upper right')
    plt.show()
    plt.plot( 'max_iter', 'MSE_train', data=table.loc[(table['activation']=='relu')&(table['learning_rate']==0.01)&(table['Hidden_layers']==2)], markersize=12, color='black', linewidth=2, label = 'relu_0.01_2')
    plt.plot( 'max_iter', 'MSE_train', data=table.loc[(table['activation']=='relu')&(table['learning_rate']==0.01)&(table['Hidden_layers']==3)], marker='', color='brown', linewidth=2, label = 'relu_0.01_3')
    plt.plot( 'max_iter', 'MSE_train', data=table.loc[(table['activation']=='relu')&(table['learning_rate']==0.1)&(table['Hidden_layers']==2)], markersize=12, color='violet', linewidth=2, label = 'relu_0.1_2')
    plt.plot( 'max_iter', 'MSE_train', data=table.loc[(table['activation']=='relu')&(table['learning_rate']==0.1)&(table['Hidden_layers']==3)], marker='', color='green', linewidth=2, label = 'relu_0.1_3')
    plt.xlabel('Iterations')
    plt.ylabel('Mean Squared Error')
    plt.legend(loc = 'upper right')
    plt.show()
    plt.plot( 'max_iter', 'MSE_test', data=table.loc[(table['activation']=='relu')&(table['learning_rate']==0.01)&(table['Hidden_layers']==2)], markersize=12, color='black', linewidth=2, label = 'relu_0.01_2')
    plt.plot( 'max_iter', 'MSE_test', data=table.loc[(table['activation']=='relu')&(table['learning_rate']==0.01)&(table['Hidden_layers']==3)], marker='', color='brown', linewidth=2, label = 'relu_0.01_3')
    plt.plot( 'max_iter', 'MSE_test', data=table.loc[(table['activation']=='relu')&(table['learning_rate']==0.1)&(table['Hidden_layers']==2)], markersize=12, color='violet', linewidth=2, label = 'relu_0.1_2')
    plt.plot( 'max_iter', 'MSE_test', data=table.loc[(table['activation']=='relu')&(table['learning_rate']==0.1)&(table['Hidden_layers']==3)], marker='', color='green', linewidth=2, label = 'relu_0.1_3')
    plt.xlabel('Iterations')
    plt.ylabel('Mean Squared Error')
    plt.legend(loc = 'upper right')
    plt.show()
    plt.plot( 'max_iter', 'MSE_train', data=table.loc[(table['activation']=='tanh')&(table['learning_rate']==0.01)&(table['Hidden_layers']==2)], markersize=12, color='yellow', linewidth=2, label = 'tanh_0.01_2')
    plt.plot( 'max_iter', 'MSE_train', data=table.loc[(table['activation']=='tanh')&(table['learning_rate']==0.01)&(table['Hidden_layers']==3)], marker='', color='purple', linewidth=2, label = 'tanh_0.01_3')
    plt.plot( 'max_iter', 'MSE_train', data=table.loc[(table['activation']=='tanh')&(table['learning_rate']==0.1)&(table['Hidden_layers']==2)], markersize=12, color='pink', linewidth=2, label = 'tanh_0.1_2')
    plt.plot( 'max_iter', 'MSE_train', data=table.loc[(table['activation']=='tanh')&(table['learning_rate']==0.1)&(table['Hidden_layers']==3)], marker='', color='olive', linewidth=2, label = 'tanh_0.1_3')
    plt.xlabel('Iterations')
    plt.ylabel('Mean Squared Error')
    plt.legend(loc='upper right')
    plt.show()
    plt.plot( 'max_iter', 'MSE_test', data=table.loc[(table['activation']=='tanh')&(table['learning_rate']==0.01)&(table['Hidden_layers']==2)], markersize=12, color='yellow', linewidth=2, label = 'tanh_0.01_2')
    plt.plot( 'max_iter', 'MSE_test', data=table.loc[(table['activation']=='tanh')&(table['learning_rate']==0.01)&(table['Hidden_layers']==3)], marker='', color='purple', linewidth=2, label = 'tanh_0.01_3')
    plt.plot( 'max_iter', 'MSE_test', data=table.loc[(table['activation']=='tanh')&(table['learning_rate']==0.1)&(table['Hidden_layers']==2)], markersize=12, color='pink', linewidth=2, label = 'tanh_0.1_2')
    plt.plot( 'max_iter', 'MSE_test', data=table.loc[(table['activation']=='tanh')&(table['learning_rate']==0.1)&(table['Hidden_layers']==3)], marker='', color='olive', linewidth=2, label = 'tanh_0.1_3')
    plt.xlabel('Iterations')
    plt.ylabel('Mean Squared Error')
    plt.legend(loc='upper right')
    plt.show()