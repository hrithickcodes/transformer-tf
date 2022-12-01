import tensorflow as tf


def get_pading_mask(source_sequence):
    padding_mask = tf.cast(tf.math.equal(source_sequence, 0), tf.float32)
    return padding_mask[:, tf.newaxis, tf.newaxis, :]  


def get_look_ahead_mask(target_sequence):
    size = tf.shape(target_sequence)[1]
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  

def get_masks(encoder_input, decoder_input):
    # padding mask to ignore the encoder paddings
    encoder_padding_mask = get_pading_mask(encoder_input)  
    # used and added with the peek ahead mask so that the decoder ignores the target paddings 
    decoder_padding_mask = get_pading_mask(decoder_input)            
    # look ahead mask so that the decoder do not learn to cheat
    decoder_peek_ahead_mask = get_look_ahead_mask(decoder_input)
    # taking the maximum of the look ahead and the decoder padding mask so that 
    # the decoder ignores the target paddings and do not look ahead to the next tokens while predicting it
    decoder_combined_mask = tf.maximum(decoder_peek_ahead_mask, decoder_padding_mask)  
    
    return encoder_padding_mask, decoder_padding_mask, decoder_combined_mask
    
    





    
