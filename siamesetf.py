#%matplotlib notebook

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import random

from pca_plotter import PCAPlotter

#print('TensorFlow version:', tf.__version__)


#Importing the data

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
#print(X_train.shape)

#normalise the by unrolling it into 784 dimensional vectors before using it
#i.e. unroll each image to a vector
X_train = np.reshape(X_train, (60000, 28*28))/255
#similarly for test set
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1]*X_test.shape[2]))/255
#print(X_train.shape)
#print(X_test.shape)


#Plotting examples

def plot_triplet(triplet):
    plt.figure(figsize=(6,2))
    for i in range(0,3):
        plt.subplot(1, 3, i+1)
        plt.imshow(np.reshape(triplet[i], (28,28)), cmap='binary')
        plt.xticks([])
        plt.yticks([])
    plt.show()

#plot_triplet([X_train[80], X_train[19], X_train[42]])

#A Batch of triplets

def create_batch(batch_size):

    #creating placeholders for anchors, positives, negatives
    anchors = np.zeros((batch_size, 784))
    positives = np.zeros((batch_size, 784))
    negatives = np.zeros((batch_size, 784))

    #populating the values
    for i in range(0, batch_size):
        #find an anchor and a positive and negative for it
        index = random.randint(0, 60000-1)
        anc = X_train[index]
        y = y_train[index]

        # find potential positive and negative indices for our anchor, gives a list
        positive_indices = np.squeeze(np.where(y_train == y))
        negative_indices = np.squeeze(np.where(y_train != y))

        pos = X_train[positive_indices[random.randint(0, len(positive_indices)-1)]]
        neg = X_train[negative_indices[random.randint(0, len(negative_indices)-1)]]

        anchors[i]=anc
        positives[i]=pos
        negatives[i]=neg

    return [anchors, positives, negatives]

#triplet = create_batch(1)
#plot_triplet(triplet)


#Embedding the model

#take images as inputs and generate n-dimensional vector representations of them
emb_dim = 64

embedding_model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation = 'relu', input_shape=(784,)),  #no. of nodes is 64
    tf.keras.layers.Dense(64, activation='sigmoid')       #output dense layer
])

#embedding_model.summary()


#what kind of emb we get from tis model
#eg = X_train[0]
#emb_eg = embedding_model.predict(np.expand_dims(eg, axis = 0))[0]
#print(emb_eg)


# Siamese Network

#create 3 inputs for anchor, positive, negative, then pass them through the
#embedding model and concatenate
inp_anc = tf.keras.layers.Input(shape=(784,))
inp_pos = tf.keras.layers.Input(shape=(784,))
inp_neg = tf.keras.layers.Input(shape=(784,))

emb_anc = embedding_model(inp_anc)
emb_pos = embedding_model(inp_pos)
emb_neg = embedding_model(inp_neg)

out = tf.keras.layers.concatenate([emb_anc, emb_pos, emb_neg], axis = 1) #axis 1 to concatenate across columns
net = tf.keras.models.Model([inp_anc, inp_pos, inp_neg],
                           out
                          )

#net.summary()

# Triplet Loss

def triplet_loss(alpha, emb_dim):
    def loss(y_true, y_pred):
        anc, pos, neg = y_pred[:, :emb_dim], y_pred[:, emb_dim:2*emb_dim], y_pred[:, 2*emb_dim:]
        dp = tf.reduce_mean(tf.square(anc-pos), axis=1)
        dn = tf.reduce_mean(tf.square(anc-neg), axis=1)
        loss_val = tf.maximum(dp-dn,0)
        return loss_val
    return loss


# Data generator

#supply stream a continuous of triplet examples
def data_generator(batch_size, emb_dim):
    while True:
        x = create_batch(batch_size)
        y = np.zeros((batch_size, 3*emb_dim))
        yield x, y


# Model Training

batch_size = 1024
epochs = 10
steps_per_epoch = int(60000/1024)

net.compile(loss = triplet_loss(alpha=0.2, emb_dim = emb_dim), optimizer = 'adam')

X,Y = X_test[:1000], y_test[:1000]


_ = net.fit(data_generator(batch_size, emb_dim),
            epochs = epochs,
            steps_per_epoch = steps_per_epoch,
            verbose = False,
            #for displaying results, shows training loss and display emb in 2D as model trains
            callbacks=[
                PCAPlotter(plt, embedding_model, X, Y)
            ]
           )
