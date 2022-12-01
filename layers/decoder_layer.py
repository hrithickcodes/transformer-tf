import tensorflow as tf
from layers.multihead_attention import Multihead_attention
from layers.pointwiseFFN import PointWiseFeedForwardNetwork

class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, 
                embedding_dimension, 
                num_attention_heads,
                ffn_units, 
                dropout_rate,
                dropout_training,
                layernorm_epsilon, 
                name = "decoder_layer"):

        super(DecoderLayer, self).__init__(name = name)
        self.embedding_dimension = embedding_dimension
        self.num_attention_heads = num_attention_heads
        self.ffn_units = ffn_units
        self.dropout_rate = dropout_rate
        self.layernorm_epsilon = layernorm_epsilon
        self.dropout_training = dropout_training


        self.self_attn_mha = Multihead_attention(self.embedding_dimension, self.num_attention_heads)
        self.cross_attn_mha = Multihead_attention(self.embedding_dimension, self.num_attention_heads)


        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=self.layernorm_epsilon)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=self.layernorm_epsilon)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=self.layernorm_epsilon)

        self.dropout1 = tf.keras.layers.Dropout(self.dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(self.dropout_rate)
        self.dropout3 = tf.keras.layers.Dropout(self.dropout_rate)

        self.ffn = PointWiseFeedForwardNetwork(self.embedding_dimension, self.ffn_units)


    def call(self, input_tensor, encoder_output, look_ahead_mask, padding_mask):

        # self-attention from the target inputs to the target inputs tokens, followed
        # by dropout and layer norm
        self_attn_out, self_attn_weight = self.self_attn_mha(input_tensor, input_tensor, input_tensor, look_ahead_mask) 
        self_attn_dropout_out = self.dropout1(self_attn_out, training=self.dropout_training)
        self_attn_layernorm1_out = self.layernorm1(tf.math.add(self_attn_dropout_out , input_tensor))

        # cross-attention between source language tokens and target language tokens
        cross_attn_out, cross_attn_weight = self.cross_attn_mha(self_attn_layernorm1_out, encoder_output, encoder_output, padding_mask)  
        cross_attn_dropout_out = self.dropout2(cross_attn_out, training=self.dropout_training)
        self_attn_layernorm2_out = self.layernorm2(tf.math.add(cross_attn_dropout_out, self_attn_layernorm1_out))  

        ffn_output = self.ffn(self_attn_layernorm2_out)  
        ffn_output = self.dropout3(ffn_output, training=self.dropout_training)
        decoderlayer_logits = self.layernorm3(tf.math.add(ffn_output , self_attn_layernorm2_out))  

        return decoderlayer_logits, self_attn_weight, cross_attn_weight

