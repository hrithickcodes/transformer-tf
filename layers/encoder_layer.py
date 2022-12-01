import tensorflow as tf
from layers.multihead_attention import Multihead_attention
from layers.pointwiseFFN import PointWiseFeedForwardNetwork

class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, 
                embedding_dimension, 
                num_attention_heads,
                ffn_units, 
                dropout_rate,
                layernorm_epsilon, 
                dropout_training,
                name = "encoder_layer"):

        super(EncoderLayer, self).__init__(name = name)
        self.embedding_dimension = embedding_dimension
        self.num_attention_heads = num_attention_heads
        self.ffn_units = ffn_units
        self.dropout_rate = dropout_rate
        self.layernorm_epsilon = layernorm_epsilon
        self.dropout_training = dropout_training


        self.mha = Multihead_attention(self.embedding_dimension, self.num_attention_heads)
        self.dropout1 = tf.keras.layers.Dropout(self.dropout_rate)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=self.layernorm_epsilon)
        self.ffn = PointWiseFeedForwardNetwork(self.embedding_dimension, self.ffn_units)
        self.dropout2 = tf.keras.layers.Dropout(self.dropout_rate)

        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=self.layernorm_epsilon)


    def call(self, input_tensor, mask = None):
        """
        Args:
        input_tensor: Sentence embeddings
        tensor of size [batch_size, sequence_length, embedding_dimension]

        output: Encoder logits of the same shape as input
        """

        # self-attention on the source language tokens
        mha_output_logits, mha_attention_scores = self.mha(input_tensor, input_tensor, input_tensor, mask) 
        mha_output_logits = self.dropout1(mha_output_logits, training=self.dropout_training)

        layernorm1_input = tf.math.add(input_tensor , mha_output_logits)
        layernorm1_out = self.layernorm1(layernorm1_input) 

        ffn_output = self.ffn(layernorm1_out)  
        ffn_output = self.dropout2(ffn_output, training=self.dropout_training)

        layernorm2_input = tf.math.add(ffn_output, layernorm1_out)
        encoderlayer_logits = self.layernorm2(layernorm2_input)  

        return encoderlayer_logits, mha_attention_scores

