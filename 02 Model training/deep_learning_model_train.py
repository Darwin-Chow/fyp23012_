import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow import keras

import os.path

from tensorflow.keras import layers

from tensorflow.keras.layers import Layer, Dense, Bidirectional, Dropout, Dense, Activation, Flatten, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential,Model,load_model
from tensorflow.keras.optimizers import SGD

from keras.layers.recurrent import LSTM

from keras.layers.merge import concatenate
from keras.utils import plot_model


from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.losses import binary_crossentropy

from sklearn.model_selection import train_test_split


# import function from another files
from model_setup import setupData, slices


from tensorflow.keras import backend as K

class RBFLayer(Layer):
    def __init__(self, units, gamma, **kwargs):
        super(RBFLayer, self).__init__(**kwargs)
        self.units = units
        self.gamma = K.cast_to_floatx(gamma)

    def build(self, input_shape):
#         print(input_shape)
#         print(self.units)
        self.mu = self.add_weight(name='mu',
                                  shape=(int(input_shape[1]), self.units),
                                  initializer='uniform',
                                  trainable=True)
        super(RBFLayer, self).build(input_shape)

    def call(self, inputs):
        diff = K.expand_dims(inputs) - self.mu
        l2 = K.sum(K.pow(diff, 2), axis=1)
        res = K.exp(-1 * self.gamma * l2)
        return res

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.units)
      
      

def main():
    
    print("\n** Setting up Data...\n")
    X, y, X_test, y_test = setupData()
    
    # load_csv_for_tensorflow()
    tensorflow_test(X, y, X_test, y_test)
    
    # deep_model_train(X, y)
    
    
    return




def test_eth_model(preprocessing_head, inputs):
  body = tf.keras.Sequential([
    layers.Dense(64),
    layers.Dense(1)
  ]) 

  preprocessed_inputs = preprocessing_head(inputs)
  result = body(preprocessed_inputs)
  model = tf.keras.Model(inputs, result)

  model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                optimizer=tf.keras.optimizers.Adam())
  return model


def load_csv_for_tensorflow():
    ethereum_data = pd.read_csv("combined_fix_bug.csv")
    
    ethereum_labels = ethereum_data['flag']
    ethereum_features = ethereum_data.drop(columns=['index', 'address', 'flag', 'token', 
                             'firstTransactionTime', 'lastTransactionTime',
                             'highestBalanceDate', 'lowestBalanceDate'
                             ])

    eth_features_dict = {name: np.array(value) 
                         for name, value in ethereum_features.items()}
    
    for example in slices(eth_features_dict):
      for name, value in example.items():
        print(f"{name:19s}: {value}")
      break
  
    inputs = {}

    for name, column in ethereum_features.items():
        dtype = column.dtype
        if dtype == object:
            dtype = tf.string
        else:
            dtype = tf.float32

    inputs[name] = tf.keras.Input(shape=(1,), name=name, dtype=dtype)
    
    numeric_inputs = {name:input for name,input in inputs.items()
                  if input.dtype==tf.float32}

    x = layers.Concatenate()(list(numeric_inputs.values()))
    norm = layers.Normalization()
    norm.adapt(np.array(ethereum_data[numeric_inputs.keys()]))
    all_numeric_inputs = norm(x)
    
    preprocessed_inputs = [all_numeric_inputs]
    
    preprocessed_inputs_cat = layers.Concatenate()(preprocessed_inputs)

    eth_preprocessing = tf.keras.Model(inputs, preprocessed_inputs_cat)
    
    tf.keras.utils.plot_model(model = eth_preprocessing , rankdir="LR", dpi=72, show_shapes=True)
    
    eth_preprocessing(eth_features_dict)
    
    

    # eth_ds = tf.data.Dataset.from_tensor_slices((eth_features_dict, ethereum_labels))
    
    # eth_batches = eth_ds.shuffle(len(ethereum_labels)).batch(32)
    
    eth_model = test_eth_model(eth_preprocessing, inputs)
    
    eth_model.fit(x=eth_features_dict, y=ethereum_labels, epochs=5)
    
    # return ethereum_labels, eth_features_dict





def deep_model_train(X, y):
    
    y = y.to_numpy()
    
    print(f"--- X shape: {X.shape}\n--- Y shape: {y.shape}\n")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=33, stratify=y)
    
    print(f"--- X_train shape: {X_train.shape}\n--- Y_train shape: {y_train.shape}\n")
    
    model = Sequential()
    
    # inputs = keras.Input(shape=(None, 64))
    
    
    # model.add(inputs)
    model.add(LSTM(10, input_shape=(10455, 64)))
    model.add(Dropout(0.5))
    model.add(Dense(800, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(200, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4, activation='softmax'))

    # model.summary()
    
    
    
    Nadam = tf.keras.optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)
    model.compile(loss='categorical_crossentropy', optimizer=Nadam, metrics=['recall'])

    history  = model.fit(X_train, y_train, verbose=1, validation_data=(X_test, y_test))

    test_loss, test_acc = model.evaluate(X_test, y_test, batch_size=64, verbose=1)
    print('Test accuracy:', test_acc)
    
    # save the model
    model.save('my_model.h5')
    
    
def tensorflow_test(eth_features, eth_labels, eth_features_test, eth_labels_test):
  
  model_path = './saved_model/ill_eth_model.keras'

  check_file_exist = os.path.isfile(model_path)
  
  if (check_file_exist):
    print("\n### Model file Exist ###\n")
    new_model = tf.keras.models.load_model(model_path)
    
    loss, acc = new_model.evaluate(eth_features_test, eth_labels_test, verbose=2)
    print('Restored model, accuracy: {:5.2f}%'.format(100 * acc))
  
  
  
  else:
    
    print("\n### Model file NOT Exist ###\n")
    print("\n### Start Building Model... ###\n")
    
    #Step 1
    model = Sequential(name='eth_model')

    #Step 2: Input layer
    model.add(layers.InputLayer(input_shape=(64,))) # necessary to use model.summary()
    # model.add(Flatten(input_shape=(64, )))
    # model.add(RBFLayer(10, 0.5))
    
    # Step 3 (2) - testing
    
    # 特徵提取層
    extract1 = LSTM(128, name='lstm1')(mnist_input)

    # 第一個解釋層
    interp1 = Dense(10, activation='relu', name='interp1')(extract1) # <-- 看這裡

    # 第二個解釋層
    interp21 = Dense(64, activation='relu', name='interp21')(extract1) # <-- 看這裡
    interp22 = Dense(32, activation='relu', name='interp22')(interp21)
    interp23 = Dense(16, activation='relu', name='interp23')(interp22)

    # 把兩個特徵提取層的結果併起來
    merge = concatenate([interp1, interp23], name='merge')

    # 輸出層
    output = Dense(10, activation='softmax', name='output')(merge)

    # 以Model來組合整個網絡
    model = Model(inputs=mnist_input, outputs=output)

    # 打印網絡結構
    model.summary()

    # plot graph
    plot_model(model, to_file='shared_feature_extractor.png')

    # 秀出網絡拓撲圖
    Image('shared_feature_extractor.png')
    

    #Step 3 (1)
    # model.add(layers.Dense(512, activation='relu'))
    # model.add(layers.Dropout(0.4)) #prevents overfitting by setting 40% of nuerons to 0
    # model.add(layers.Dense(512, activation='relu'))
    # model.add(layers.Dropout(0.4))
    # model.add(layers.Dense(128, activation='relu'))
    # model.add(layers.Dense(1, activation='sigmoid')) # output layer, use sigmoid for binary

    # model.summary()
    
    #Step 4
    model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(0.0001), metrics=['accuracy'])
    
    class myCallback(tf.keras.callbacks.Callback):
      def on_epoch_end(self, epoch, logs={}):
          if(logs.get('val_accuracy')>0.99):
              print("\nReached 99% accuracy so cancelling training!")
              self.model.stop_training = True


    #Step 6
    history = model.fit(eth_features, eth_labels, 
                        validation_data=(eth_features_test, eth_labels_test), 
                        batch_size=8, 
                        callbacks=[myCallback()],
                        epochs=50)
    
    probability_model = tf.keras.Sequential([
      model,
      tf.keras.layers.Softmax()
    ] )
    
    probability_model(eth_features_test [:5])
    
    # Save the entire model as a `.keras` zip archive.
    model.save(model_path)
  
  # print("nTest Accuracy: {0:f}n".format(accuracy_score))
  
  
  
  # eth_features = np.array(eth_features)
  
  # model = tf.keras.Sequential([
  #   layers.Dense(64),
  #   layers.Dense(1, activation='sigmoid')
  # ])

  # model.compile(loss = tf.losses.MeanSquaredError(),
  #                     optimizer = tf.optimizers.Adam(),
  #                     metrics=['accuracy'])
  
  # model.fit(eth_features, eth_labels, epochs=20)
  
  
  # loss, acc = model.evaluate(eth_features_test,  eth_labels_test, verbose=2)
  
  # probability_model = tf.keras.Sequential([
  #   model,
  #   tf.keras.layers.Softmax()
  # ] )
  
  # probability_model(eth_features_test [:5])
  
  


if __name__ == "__main__":
    main()