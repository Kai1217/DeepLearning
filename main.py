"
This is an application made for my learning of creating deep learning models in the Python deep learning framework
called Keras.
"
import keras
import tensorflow as tf

class SimpleMLP(keras.Model):

    def __init__(self, use_bn=False, use_dp=False, num_classes=10):
        super(SimpleMLP, self).__init__(name='mlp')
        self.use_bn = use_bn
        self.use_dp = use_dp
        self.num_classes = num_classes
        
        self.dense1 = keras.layers.Dense(32, activation='relu')
        self.dense2 = keras.layers.Dense(num_classes, activation='softmax')
        if self.use_dp:
            self.dp = keras.layers.Droupout(0.5)
        if self.use_bn:
            self.bn = keras.layers.BatchNormalization(axis=-1)

        def call(self, inputs):
            x = self.dense1(inputs)
            if self.use_dp:
                x = self.dp(x)
            if self.use_bn:
                x = self.bn(x)
            return self.dense2(x)

model = SimpleMLP()
model.compile(...)
model.fit(...)
