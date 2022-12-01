import tensorflow as tf
from utils.positional_encoding import positional_encoding
from layers.decoder_layer import DecoderLayer


class Decoder(tf.keras.layers.Layer):
    def __init__(self, 
                target_vocab_size,
                embedding_dimension,
                number_of_attention_heads,
                num_stacked_decoders,
                ffn_units,
                dropout_rate,
                dropout_training,
                layernorm_epsilon,
                name = "Decoder"):
        super(Decoder, self).__init__(name = name)

        self.target_vocab_size = target_vocab_size
        self.embedding_dimension = embedding_dimension
        self.number_of_attention_heads = number_of_attention_heads
        self.num_stacked_decoders = num_stacked_decoders
        self.ffn_units = ffn_units
        self.dropout_rate = dropout_rate
        self.dropout_training = dropout_training
        self.layernorm_epsilon = layernorm_epsilon
        
        
        self.target_embedding_layer = tf.keras.layers.Embedding(self.target_vocab_size, self.embedding_dimension)
        self.decoder_pos_encodings = positional_encoding(self.target_vocab_size, self.embedding_dimension)
        
        self.stacked_decoder_layers = [DecoderLayer(embedding_dimension = self.embedding_dimension, 
                                                    num_attention_heads = self.number_of_attention_heads,
                                                    ffn_units = self.ffn_units,
                                                    dropout_rate = self.dropout_rate,
                                                    dropout_training = self.dropout_training,
                                                    layernorm_epsilon=self.layernorm_epsilon,
                                                    name = f"decoder_{dec_name + 1}") for dec_name in range(self.num_stacked_decoders)]
        self.dropout = tf.keras.layers.Dropout(self.dropout_rate)
        
    def call(self, decoder_input, encoder_output, peek_ahead_mask, decoder_padding_mask):
        decoder_target_seq_length = tf.shape(decoder_input)[1]
        
        attention_weights = {}

        decoder_input_embeddings = self.target_embedding_layer(decoder_input)  
        decoder_input_embeddings *= tf.math.sqrt(tf.cast(self.embedding_dimension, tf.float32))

        
        decoder_input_embeddings += self.decoder_pos_encodings[:, :decoder_target_seq_length, :]
        decoder_input_embeddings = self.dropout(decoder_input_embeddings,
                                                training = self.dropout_training)

        for i in range(self.num_stacked_decoders):
            decoder_input_embeddings, self_attention_scores, cross_attention_scores = self.stacked_decoder_layers[i](decoder_input_embeddings,
                                                                                                                    encoder_output,
                                                                                                                    peek_ahead_mask,
                                                                                                                    decoder_padding_mask)

            attention_weights['decoder_layer_self_attention_{}'.format(i + 1)] = self_attention_scores
            attention_weights['decoder_layer_cross_attention_{}'.format(i + 1)] = cross_attention_scores

        return decoder_input_embeddings, attention_weights
        
        