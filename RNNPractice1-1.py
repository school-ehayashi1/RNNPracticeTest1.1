
import tensorflow as tf
from tensorflow import keras
import numpy as np
import re 
train = pd.read_csv("train.txt")
test = pd.read_csv("test.txt")

train = train.tolist()
train = re.sub('\s+', '', train)

# potential ordinal encoder for preprocessing the text into encoding for model
from sklearn.preprocessing import OrdinalEncoder
onc = OrdinalEncoder(dtype="int64", categories="auto")
onc.fit_transform("")
model = keras.Sequential() 

# Note to self, quoted from tensorflow.org: By default, the output of a RNN layer contains a single vector per sample. This vector is the RNN cell output corresponding to the last timestep, containing information about the entire input sequence. The shape of this output is (batch_size, units) where units corresponds to the units argument passed to the layer's constructor.
#A RNN layer can also return the entire sequence of outputs for each sample (one vector per timestep per sample), if you set return_sequences=True. The shape of this output is (batch_size, timesteps, units).
# Basically rnn without returnsequences return batch_size, units
# With returns batch_size, timestep, units

model.add(keras.layers.Embedding(input_dim = uniq, output_dim = 256))
model.add(keras.layers.GRU(64, return_sequences=True))
