import tensorflow as tf
from layers.self_attention import Scaled_dot_product_attention

class Multihead_attention(tf.keras.layers.Layer):
    def __init__(self, embedding_dimension, number_of_heads, name = "Multihead_attention"):
        super(Multihead_attention, self).__init__(name = name)
        self.embedding_dimension = embedding_dimension
        self.number_of_heads = number_of_heads
        
        self.head_dimension = self.embedding_dimension // self.number_of_heads
    
        assert (self.head_dimension * self.number_of_heads) == self.embedding_dimension , \
            "Embedding dimension should be divisible by the numbe of heads"

        self.wQ = tf.keras.layers.Dense(self.embedding_dimension, 
                                        use_bias=False)
        self.wK = tf.keras.layers.Dense(self.embedding_dimension,
                                        use_bias=False)
        self.wV = tf.keras.layers.Dense(self.embedding_dimension,
                                        use_bias=False)

        self.FFN = tf.keras.layers.Dense(self.embedding_dimension)

        self.sda = Scaled_dot_product_attention()

    def split_to_heads(self, input_tensor, batch_size):
        """
        Args: 
        input_tensor | shape: [batch size, sequence_length, embedding_dimension]
        batch_size is batch size
        output:
        input_tensor | shape: [batch size, num_attention_heads, sequence_length, head dimension]

        """
        splitted_shape = (batch_size, -1, self.number_of_heads, self.head_dimension)
        splitted_input_tensor = tf.reshape(input_tensor, splitted_shape)
        return tf.transpose(splitted_input_tensor, perm=[0, 2, 1, 3])


    def call(self, queries, keys, values, mask = None):
        self.mask = mask
        self.batch_size = tf.shape(queries)[0]
        self.batch_size = tf.cast(self.batch_size, tf.int64)
        
        # learning the query, key and value matrices
        queries = self.wQ(queries) 
        keys = self.wK(keys)  
        values = self.wV(values)


        # splitting the last embedding dimension with number_of_heads, head_dimension
        # shape of all the tensors: [batch_size, num_attention_heads, sequence_length, head_dimension]
        self.queries = self.split_to_heads(queries, self.batch_size)
        self.keys = self.split_to_heads(keys, self.batch_size)
        self.values = self.split_to_heads(values, self.batch_size)

        # Computing logits and attention weights using scaled dot product attention
        logits, attention_weights = self.sda(self.queries,
                                            self.keys,
                                            self.values,
                                            self.mask)

        # Earlier [batch size, numbe of attention heads, sequence length, head_dimension]
        # after transpose: [batch_size, sequence length, num_attention heads, head_dimenson]
        logits = tf.transpose(logits, perm=[0, 2, 1, 3])  
        # number_of_heads, head_dimension -> embedding_dimension, concating the logits
        # concatenating to [batch_size, sequence length, num_attention heads * head_dimenson]
        concated_logits = tf.reshape(logits,
                                      (self.batch_size, -1, self.embedding_dimension)) 

        output_logits = self.FFN(concated_logits)  

        return output_logits, attention_weights




