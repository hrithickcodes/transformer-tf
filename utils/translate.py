import numpy as np
import tensorflow as tf
from utils.utils import get_transformer_logits, get_input_ids_from_texts, save_attention_plots

def Translate(model, 
            source_language,
            source_tokenizer,
            target_tokenizer,
            config,
            plot_attention_weights = False):

    # getting the input ids for the source language
    source_input_ids = get_input_ids_from_texts([source_language], source_tokenizer)
        # the start and end token ids for the decoder
    target_start_token_id = len(target_tokenizer.word_index)
    target_end_token_id = len(target_tokenizer.word_index) + 1
    
    # the decoder is autoregressive so the first input is the start token
    decoder_target_ids = tf.fill(value = target_start_token_id, dims = (source_input_ids.shape[0], 1))
    
    # decoder_prediction_length <= maximum_length
    for i in range(config["model_architecture"]["max_sentence_length"]):
        # getting the decoder output logits and attention scores dict
        transformer_logits, attention_scores_dict = get_transformer_logits(model, 
                               source_input_ids,
                               decoder_target_ids,
                               return_attentions = True)
        transformer_logits = transformer_logits[:, -1:, :]
         
        # getting the predicted id that has the highest probability
        predicted_id = tf.cast(tf.argmax(transformer_logits, axis=-1), tf.int32) 
        
        # the for loop will break once the decoder outputs the end_token
        if tf.equal(tf.squeeze(predicted_id), target_end_token_id):
            decoder_target_ids = tf.concat([decoder_target_ids, predicted_id], axis=-1)
            decoder_target_ids = tf.squeeze(decoder_target_ids, axis=0)
            predicted_translation = target_tokenizer.sequences_to_texts([decoder_target_ids.numpy()[1:-1]])
            predicted_translation = tf.squeeze(predicted_translation).numpy()
            predicted_translation = str(predicted_translation,'utf-8')
            
            if plot_attention_weights:
                save_attention_plots(attention_scores_dict, 
                               source_language,
                               predicted_translation,
                               config["inference_settings"]["visualise_attention_layer"],
                               config["inference_settings"]["attention_plot_save_dir"])
                
            return predicted_translation, attention_scores_dict

        # concatenating the predicted id with the decoder input and again it will be passed inside
        decoder_target_ids = tf.concat([decoder_target_ids, predicted_id], axis=-1)
        
    decoder_target_ids = tf.squeeze(decoder_target_ids, axis=0)    
    predicted_translation = target_tokenizer.sequences_to_texts([decoder_target_ids.numpy()[1:]])
    predicted_translation = tf.squeeze(predicted_translation)
    
    if plot_attention_weights:
        save_attention_plots(attention_scores_dict, 
                        source_language,
                        predicted_translation,
                        config["inference_settings"]["visualise_attention_layer"])
        
    return predicted_translation, attention_scores_dict
        
    