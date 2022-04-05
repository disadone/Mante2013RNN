import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Layer
from sklearn.model_selection import train_test_split

path='./data/mante2013_u'
dataset = tf.data.experimental.load(path)
dataset=dataset.shuffle(10000)
num_train=8000
train_dataset=dataset.take(num_train).batch(20,drop_remainder=True)
valid_dataset=dataset.skip(num_train).batch(20,drop_remainder=True)
# for x,y in dataset.take(1):
#     print(x.shape,y.shape)

dt=1
class ManteCell(Layer): # 里面定义ODE方程，但是怎么把input放进去呢
    def __init__(self, units, **kwargs):
        self.units=units
        self.state_size=(units,)
        self.output_size=(units,)
        super(ManteCell, self).__init__(**kwargs)
    
    def build(self, input_shape): # 将原来的参数设为可训练的参数
        self.J = self.add_weight(shape=(self.units,self.units),
                                 initializer="random_normal",
                                 trainable=True)
        self.B = self.add_weight(shape=(self.units,input_shape[-1]),
                                 initializer="random_normal",
                                 trainable=True)
        self.c_x = self.add_weight(shape=(self.units,),
                                 initializer='zeros',
                                 trainable=True)
    # def get_initial_state(self,inputs=None, batch_size=None, dtype=None):
        self.built = True

    def call(self,u_t,states):
        x=states[0]
        rou_x = tf.random.normal((1,),mean=0.,stddev=0.1)

        r=tf.math.tanh(x)
        new_x = x + tf.linalg.matvec(self.J,r) + tf.linalg.matvec(self.B,u_t) + self.c_x + rou_x
        
        return new_x, [new_x]


from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import SimpleRNN,Dense,RNN,Input,Activation
from tensorflow.keras.activations import tanh


# model=Sequential([
#     SimpleRNN(100,input_shape=(750,4),activation='linear'),
#     Dense(1,activation='linear')
# ])

inputs=Input(shape=(750,4))
x=RNN(ManteCell(100),time_major = False)(inputs)
x=Activation(tanh)(x)
outputs=Dense(1,activation='linear')(x)
model=Model(inputs,outputs)


optimizer = tf.keras.optimizers.Adam(1e-4)
model.compile(optimizer=optimizer, loss='mse')
model.fit(train_dataset,validation_data=valid_dataset,epochs=2)

weights=model.get_weights()
import numpy as np
np.save("data/weights.npy",weights)
...