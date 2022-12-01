import tensorflow as tf
from layers.encoder_layer import EncoderLayer
from utils.positional_encoding import positional_encoding

class Encoder(tf.keras.layers.Layer):
    def __init__(self, 
                source_vocab_size,
                embedding_dimension,
                number_of_attention_heads,
                num_stacked_encoders,
                ffn_units,
                dropout_rate,
                dropout_training,
                layernorm_epsilon,
                name = "Encoder"):
        
        super(Encoder, self).__init__(name = name)
        
        self.source_vocab_size = source_vocab_size
        self.embedding_dimension = embedding_dimension
        self.number_of_attention_heads = number_of_attention_heads
        self.num_stacked_encoders = num_stacked_encoders
        self.ffn_units = ffn_units
        self.dropout_rate = dropout_rate
        self.dropout_training = dropout_training
        self.layernorm_epsilon = layernorm_epsilon


        self.source_embedding_layer = tf.keras.layers.Embedding(self.source_vocab_size,
                                                                self.embedding_dimension)

        self.positional_encoding = positional_encoding(self.source_vocab_size, self.embedding_dimension)

            
        self.stacked_encoder_layers = [EncoderLayer(embedding_dimension = self.embedding_dimension, 
                                                    num_attention_heads = self.number_of_attention_heads,
                                                    ffn_units = self.ffn_units,
                                                    dropout_rate=self.dropout_rate,
                                                    layernorm_epsilon = self.layernorm_epsilon,
                                                    dropout_training = self.dropout_training,
                                                    name = f"encoder_{en_name}") for en_name in range(self.num_stacked_encoders)]

        self.dropout = tf.keras.layers.Dropout(self.dropout_rate)
        
        self.attention_dict = {}
        
    def call(self, encoder_input, mask = None):
        """
        Encoders only perform self-attention on the input sentence.

        Parameters:
        ----------
        encoder input: Tensor
            The source sentence embeddings
            shape: [batch_size, sequence_length]  

        Returns:
        ----------
        Tensor of shape [batch_size, sequence_length, embedding_dimension]

        """

        # getting the sequence length
        encoder_seq_len = tf.shape(encoder_input)[1]

        # passing it through embedding layer
        embeddings = self.source_embedding_layer(encoder_input) 
        # scaling the embeddings using the embedding dimension
        embeddings *= tf.math.sqrt(tf.cast(self.embedding_dimension, tf.float32))

        # adding positional encodings
        embeddings += self.positional_encoding[:, :encoder_seq_len, :]

        # dropout after adding positional encodings
        out = self.dropout(embeddings,
                           training = self.dropout_training)
        
        # passing through the stack of encoders
        for i in range(self.num_stacked_encoders):
            out, encoder_self_attention = self.stacked_encoder_layers[i](out, mask)
            self.attention_dict[f"encoder_block{i+1}_attention"] = encoder_self_attention

        
        return out

        


                                                                
