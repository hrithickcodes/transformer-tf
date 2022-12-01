import os
import pandas as pd
import seaborn as sns
import numpy as np
import tensorflow as tf
from utils.masking import get_masks
import matplotlib.pyplot as plt    


def get_transformer_logits(model,
                            encoder_input, 
                            decoder_input,
                            return_attentions = False):
    encoder_padding_mask, decoder_padding_mask, decoder_combined_mask = get_masks(encoder_input, decoder_input)
    output_logits, attention_scores_dict = model(encoder_input,
                                                decoder_input,
                                                encoder_padding_mask,
                                                encoder_padding_mask,
                                                decoder_combined_mask,
                                                training = False)
    if return_attentions:
        return output_logits, attention_scores_dict
    return output_logits


class CustomLRForAdam(tf.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, embedding_dimension, warmup_steps=4000):
    super(CustomLRForAdam, self).__init__()

    self.embedding_dimension = tf.cast(embedding_dimension, tf.float32)
    self.warmup_steps = warmup_steps

  def __call__(self, step):
    half_power = tf.cast(-0.5, tf.float32)
        
    step_var = tf.math.pow(x = step, y = half_power)
    
    warmup_epoch_power = tf.cast(-1.5, tf.float32)
    warmup_epoch_var = tf.math.pow(x = self.warmup_steps, y = warmup_epoch_power)

    first_term = tf.math.pow(x = self.embedding_dimension, y = half_power)
    second_term = tf.math.minimum(step_var, step * warmup_epoch_var)
    
    LRs = first_term * second_term
    return LRs

def get_input_ids_from_texts(texts, tokenizer, labels = False):
    input_ids = tf.ragged.constant(tokenizer.texts_to_sequences(texts))
    source_start_token, source_end_token = len(tokenizer.word_index), len(tokenizer.word_index) + 1
    start_toks = tf.fill(dims = (input_ids.shape[0], 1), value = source_start_token)
    end_toks = tf.fill(dims = (input_ids.shape[0], 1), value = source_end_token)
    if labels:
        return  tf.concat([input_ids, end_toks],axis=1).to_tensor()
    return  tf.concat([start_toks, input_ids, end_toks],axis=1).to_tensor()


def save_attention_plots(attention_dict, source_text, target_text, vis_attention_name, save_dir):
    sns.set_style("dark")
    retrived_attention_matrix = attention_dict[vis_attention_name]
    retrived_attention_matrix = tf.squeeze(retrived_attention_matrix, axis = 0)
    num_attention_heads = retrived_attention_matrix.shape[0]
    for head in range(num_attention_heads):
        attention_head_matrix = retrived_attention_matrix[head]
        attention_df = pd.DataFrame(attention_head_matrix)
        attention_df.columns = ["START"] + source_text.split(" ") + ["END"]
        attention_df.index = target_text.split(" ") + ["END"]
        shm = sns.heatmap(attention_df, annot=True)
        figure = shm.get_figure()    
        figure.savefig(os.path.join(save_dir, f"attention_head_{head + 1}.jpg"), dpi=400)
        plt.tight_layout()
        plt.clf()
        
def plot_loss(epochdict, savedir, show = True):
    training_losses, test_losses = epochdict["Training Loss"], epochdict["Test Loss"]
    
    plt.grid(True)
    plt.plot(np.arange(len(training_losses)),training_losses, color = "blue")
    plt.plot(np.arange(len(test_losses)), test_losses, color = "orange")
    plt.legend(['Training Loss', 'Test Loss'])
    plt.title('Loss vs Epoch')
    if savedir:
        savepath = os.path.join(savedir, "train-test-loss.png")
        plt.savefig(savepath)
    
    if show:    plt.show()
    plt.clf()
    print(f"Loss results saved at {savepath}")
    
    
def plot_accuracy(epochdict, savedir, show = True):
    training_accs, test_accs = epochdict["Training Accuracy"], epochdict["Test Accuracy"]
    
    plt.grid(True)
    plt.plot(np.arange(len(training_accs)),training_accs, color = "blue")
    plt.plot(np.arange(len(test_accs)), test_accs, color = "orange")
    plt.legend(['Training Accuracy', 'Test Accuracy'])
    plt.title('Accuracy vs Epoch')
    if savedir:
        savepath = os.path.join(savedir, "train-test-accuracy.png")
        plt.savefig(savepath)
    
    if show:    plt.show()
    plt.clf()
    print(f"Accuracy results saved at {savepath}")
    
def plot_learning_rate(learning_rates, savedir, show = True):
    plt.grid(True)
    plt.plot(learning_rates)
    plt.ylabel("Learning Rates")
    plt.xlabel("Train steps")
    plt.title("Learning rate vs Training steps")
    if savedir:
        savepath = os.path.join(savedir, "learning_rate.png")
        plt.savefig(savepath)
    
    if show:    plt.show()
    plt.clf()
    print(f"learning_rate plot saved at {savepath}")
   


