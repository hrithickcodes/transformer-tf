import tensorflow as tf
from encoder import Encoder
from decoder import Decoder

class Transformer(tf.keras.Model):
    def __init__(self, 
                 source_vocab_size,
                 target_vocab_size,
                 number_of_layers,
                 embedding_dimension,
                 number_of_attention_heads,
                 ffn_units,
                 dropout_rate,
                 dropout_training,
                 layernorm_epsilon,
                 name = "Transformer"):
        super(Transformer, self).__init__(name = name)
        
        self.source_vocab_size = source_vocab_size
        self.target_vocab_size = target_vocab_size
        self.number_of_layers = number_of_layers
        self.embedding_dimension = embedding_dimension
        self.number_of_attention_heads = number_of_attention_heads
        self.ffn_units = ffn_units
        self.dropout_rate = dropout_rate
        self.dropout_training = dropout_training
        self.layernorm_epsilon = layernorm_epsilon
        
        self.encoders =  Encoder(source_vocab_size = self.source_vocab_size,
                                embedding_dimension = self.embedding_dimension,
                                number_of_attention_heads = self.number_of_attention_heads,
                                num_stacked_encoders =  self.number_of_layers,
                                ffn_units = self.ffn_units,
                                dropout_rate = self.dropout_rate, 
                                dropout_training = self.dropout_training,
                                layernorm_epsilon = self.layernorm_epsilon)  
        
        self.decoders =  Decoder(target_vocab_size = self.source_vocab_size,
                                embedding_dimension = self.embedding_dimension,
                                number_of_attention_heads = self.number_of_attention_heads,
                                num_stacked_decoders =  self.number_of_layers,
                                ffn_units = self.ffn_units,
                                dropout_rate = self.dropout_rate, 
                                dropout_training = self.dropout_training,
                                layernorm_epsilon = self.layernorm_epsilon)  
        
        self.output_layer = tf.keras.layers.Dense(self.target_vocab_size)  
        
        

    def call(self, source, target, encoder_padding_mask, decoder_padding_mask, peek_ahead_mask, training = False):
        encoder_logits = self.encoders(encoder_input = source,
                                       mask = encoder_padding_mask)
        
        decoder_logits, attention_scores_dict = self.decoders(target,
                                                              encoder_logits,
                                                              peek_ahead_mask,
                                                              decoder_padding_mask)
        
        decoder_logits = self.output_layer(decoder_logits)
        
        return tf.nn.softmax(decoder_logits, axis = -1), attention_scores_dict
        
        