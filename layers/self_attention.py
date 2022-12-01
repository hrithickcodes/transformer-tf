import tensorflow as tf

class Scaled_dot_product_attention(tf.keras.layers.Layer):
    """
    Computes scale dot product attention, this class can be used for computing
    Global attention and causal attention.
    """
    def __init__(self, name =  "scaled_product_attention"):
        super(Scaled_dot_product_attention, self).__init__(name = name)
        

    def call(self, query, key, value, mask = None, e = -1e9):
        self.query = query
        self.key = key
        self.value = value

        # Matrix multiplication of the Query and the keys
        # shape: [batch size, number of attention heads, sequence length, sequnce length]
        Q_K_matmul = tf.matmul(a = self.query,
                                b = self.key,
                                transpose_b= True )

        # getting the embedding dimension from the keys
        self.d = tf.shape(self.key)[-1]
        self.d = tf.cast(self.d, tf.float32)
    
        # scaling the matrix with the embedding dimension for stable gradients
        raw_scores = tf.divide(Q_K_matmul, tf.math.sqrt(self.d))

        # if we have mask then multiplying -inf with the mask so that the softmax values become zero.
        if mask is not None:
            raw_scores += tf.multiply(mask, e)

        # comuting the softmax with respect to the last dimension
        # last sequence length axis 
        # shape: [batch size, numbe of attention heads, sequence length, sequence length (softmax axis)]
        attention_weights = tf.nn.softmax(raw_scores, axis = -1)

        # weighing the attention score with the values
        # shape: [batch size, numbe of attention heads, sequence length, head_dimension]
        output_logits = tf.matmul(attention_weights, self.value)

        return output_logits, attention_weights

        





        