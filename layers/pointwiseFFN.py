from re import M
import tensorflow as tf

class PointWiseFeedForwardNetwork(tf.keras.layers.Layer):
    def __init__(self,
                embedding_dimension, 
                ffn_units,
                kernel_initializer = tf.keras.initializers.HeNormal(),
                name = "pointWise_feed_forward_network"):
        super(PointWiseFeedForwardNetwork, self).__init__(name = name)
        self.embedding_dimension = embedding_dimension
        self.ffn_units = ffn_units

        self.ffn1 = tf.keras.layers.Dense(self.ffn_units,
                                        activation="relu",
                                        use_bias=False,
                                        kernel_initializer=kernel_initializer)

        self.ffn2 = tf.keras.layers.Dense(self.embedding_dimension,
                                          use_bias=False,
                                         kernel_initializer=kernel_initializer)

    def call(self, input_tensor):
        self.ffn1_out = self.ffn1(input_tensor)
        self.ffn2_output = self.ffn2(self.ffn1_out)
        return self.ffn2_output

    