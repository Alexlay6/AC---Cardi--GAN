# -*- coding: utf-8 -*-
"""
Created on Thu Dec  2 14:42:50 2021

@author: alexj
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tqdm import tqdm
import pandas as pd


import matplotlib.pyplot as plt
import numpy as np

Y_real = pd.read_csv('Y_all_diagnostics.csv')
Y_real=Y_real[['NORM', 'STTC', 'HYP', 'CD', 'MI']]
Y_real=np.array(Y_real)



BATCH_SIZE = 200
def real_samples():  
   
    X_real = np.loadtxt('X_all_diagnostics.csv')
    X_real = X_real.reshape(118967,200,1)
    X_real = X_real[:1000]
    Y_real = pd.read_csv('Y_all_diagnostics.csv')
    Y_real=Y_real[['NORM', 'STTC', 'HYP', 'CD', 'MI']]
    Y_real=np.array(Y_real)
    Y_real = Y_real[:1000]
    
    train_dataset = train_dataset = tf.data.Dataset.from_tensor_slices((X_real,Y_real)).shuffle(200000).batch(BATCH_SIZE)
    return train_dataset 
data = real_samples()
                                                  


def define_generator(inputs, labels, len_data):
    
    inputs = [inputs, labels]
    x = layers.concatenate(inputs, axis = 1)
    
    
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)

    x = layers.Conv1D(filters=128, kernel_size=16, strides=1, padding='same')(x)
    x = layers.ReLU()(x)
  
    x = layers.Conv1D(filters=64, kernel_size=16, strides=1, padding='same')(x)
    x = layers.ReLU()(x)
    
    x = layers.UpSampling1D(2)(x)
    
    x = layers.Conv1D(filters=32, kernel_size=16, strides=1, padding='same')(x)
    x = layers.ReLU()(x)
    
    x = layers.Conv1D(filters=16, kernel_size=16, strides=1, padding='same')(x)
    x = layers.ReLU()(x)

    x = layers.UpSampling1D(2)(x)
    
    x = layers.Conv1D(filters=1, kernel_size=16, strides=1, padding='same', activation='sigmoid')(x)
    

    generator = keras.Model(inputs , x, name = 'Generator')
    return generator

def define_discriminator(inputs):
    
    
    x = inputs
    
    x = layers.Conv1D(filters=32, kernel_size=16, strides=1, padding='same')(x)
    x = layers.LeakyReLU()(x)

    x = layers.Dropout(0.4)(x)

    x = layers.Conv1D(filters=64, kernel_size=16, strides=1, padding='same')(x)
    x = layers.LeakyReLU()(x)

    x = layers.MaxPool1D(pool_size=2)(x)

    x = layers.Conv1D(filters=128, kernel_size=16, strides=1, padding='same')(x)
    x = layers.LeakyReLU()(x)

    x = layers.Dropout(0.4)(x)

    

    x = layers.MaxPool1D(pool_size=2)(x)

    x = layers.Flatten()(x)
    
    outputs = layers.Dense(1)(x)#check this shouldnt be sigmoid
    label = layers.Dense(128)(x)
    label = layers.Dense(5)(label)
    label = layers.Activation('softmax', name = 'label')(label)
    
    outputs = [outputs, label]
    discriminator = keras.Model(inputs, outputs, name = 'Discriminator')
    return discriminator

    discriminator, generator, gan = models
    batch_size, latent_size, n_steps, name_model, num_labels, n_epochs = params
    save_epoch = 1
    
    disc_loss_ = []
    gen_loss_ = []
    number_epochs = []
    for epoch in tqdm(range(n_epochs)):
        
        for beat in data:
            real_data = beat[0]
        
            batch_size = tf.shape(real_data)[0]
        
            real_labels = beat[1]
            noise = np.random.uniform(-1.0,1.0,size=[batch_size, latent_size]) 
           
            
            fake_labels = np.eye(num_labels)[np.random.choice(num_labels, batch_size)]
            fake_data = generator.predict([noise, fake_labels])
            
            x = np.concatenate((real_data,fake_data))
            labels = np.concatenate((real_labels,fake_labels))
            y = np.ones([2*batch_size, 1])
            y[batch_size:,:] = 0
           
            metrics = discriminator.train_on_batch(x,[y,labels])
            
            
            log = "%d: [Discriminator loss: %f, Activation loss: %f, Label loss: %f, Activation accuracy: %f, Label Accuracy: %f ]" % (epoch , metrics[0], metrics[1], metrics[2], metrics[3], metrics[4])
            y = np.ones([batch_size, 5])
            metrics_ = gan.train_on_batch([noise, fake_labels], [y, fake_labels])
        
            log = "%s: [GAN loss: %f, Activation loss: %f, Label loss: %f, Activation accuracy: %f, Label Accuracy: %f" % (log, metrics_[0], metrics_[1], metrics_[2], metrics_[3], metrics_[4])
        disc_loss_.append(metrics[0])
        gen_loss_.append(metrics_[0])
        number_epochs.append(epoch+1)
        print('Epoch number: ', epoch+1)
        print(log)
        if (epoch+1) % save_epoch == 0:
            noise_plot = np.random.uniform(-1.0,1.0,size=[1, latent_size])
            plot_labels = np.eye(num_labels)[np.random.choice(num_labels, 1)]
        
            
            x = generator.predict([noise_plot, plot_labels])
            
            label_pred = (discriminator.predict([x, plot_labels]))
            print('Label given: ', plot_labels)
            print('Prediction of label: ', label_pred)
            
       
            plot_ex=beat[0].numpy()
            num = np.random.randint(0, 100)
            plot_ex = plot_ex[num]
        
            c = np.array(range(200))
            c = c.reshape(200, 1)
            x = x.reshape(200,1)
            plot_ex = plot_ex.reshape(200,1)
            X = np.hstack((c, x))
            Y = np.hstack((c,plot_ex))
            fig, axs = plt.subplots(2)
            axs[0].plot(X[:,0], X[:,1], color = 'blue')
            axs[0].plot(Y[:,0], Y[:,1], color = 'red')
            axs[0].set_title('Epoch Number: {}'.format(epoch))
            axs[0].set_xlabel('Time interval')
            axs[0].set_ylabel('Normalised data value')
              
              
        
            axs[1].plot(number_epochs, gen_loss_, color = 'g', label = 'Gen loss')
            axs[1].plot(number_epochs, disc_loss_, color = 'c', label = 'Disc loss')
            axs[1].legend(loc="upper left")
            axs[1].set_xlabel('Epoch number')
            axs[1].set_ylabel('Loss')
            
            fig.savefig("Images_for_ACGAN/Image_AFIB{}".format(epoch))
            plt.close()
        
    
    generator.save(name_model + ".h5")
def build_train():
    
    
    
    name_model = "ACGAN"
    latent_size = 45
    latent_dim = 1
    num_labels = 5
    batch_size = BATCH_SIZE
    len_data = 200
    n_epochs = 100
    n_steps = 118967
    lr = 2e-4
    decay = 6e-6
    label_shape = (num_labels, 1)
    print(label_shape )
    input_shape = (200,1)
    inputs = keras.Input(shape=input_shape, name = 'discriminator_input')
    labels = keras.Input(shape=label_shape, name='class_labels')
    loss = ['binary_crossentropy', 'binary_crossentropy']
    discriminator = define_discriminator(inputs)
    op = tf.keras.optimizers.RMSprop(lr=lr, decay = decay)
    discriminator.compile(loss= loss, optimizer= op, metrics = ['accuracy'])
    discriminator.summary()
    input_shape = (latent_size, 1)
    inputs = keras.Input(shape = input_shape, name = 'z_input')
    generator = define_generator(inputs, labels , len_data)
    
    generator.summary()
    optimizer = tf.keras.optimizers.RMSprop(lr=lr*0.5 , decay = decay*0.5)
    discriminator.trainable = False
    outputs = discriminator(generator([inputs, labels]))
    gan = keras.Model([inputs, labels], outputs, name=name_model)
    gan.compile(loss= loss, optimizer = optimizer, metrics = ['accuracy'])
    gan.summary()
    from tensorflow.keras.utils import plot_model
    plot_model(gan, to_file='model.png')
    models = (discriminator, generator, gan)
    params = (batch_size, latent_size, n_steps, name_model, num_labels, n_epochs)
    train(models, data, params)
    
    
build_train()

