import tensorflow as tf

def MaskedCCEloss(y_true, y_pred):
    # masking the padding tokens so that we do not calculate the loss regarding padding tokens
    mask = tf.math.logical_not(tf.math.equal(y_true, 0))
    CCEloss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False,
                                                            reduction="none")
    loss_ = CCEloss(y_true, y_pred)
    mask = tf.cast(mask, dtype=loss_.dtype)
    # discarding the padding tokens losses
    loss_ *= mask
    return tf.reduce_mean(loss_)
