import tensorflow as tf
from utils.loss import MaskedCCEloss
from utils.performance_metrics import masked_accuracy
from utils.masking import get_masks

@tf.function
def trainer(model,
            encoder_input, 
            decoder_input, 
            labels,
            optimizer):

    with tf.GradientTape() as tape:
        encoder_padding_mask, decoder_padding_mask, decoder_combined_mask = get_masks(encoder_input, decoder_input) 
        output_logits, attention_scores_dict = model(encoder_input,
                                                    decoder_input,
                                                    encoder_padding_mask,
                                                    encoder_padding_mask,
                                                    decoder_combined_mask,
                                                    training = True)
        loss = MaskedCCEloss(labels, output_logits)
        
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    return loss, masked_accuracy(labels, output_logits)

    
                            
    
