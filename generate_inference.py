import numpy as np
import tensorflow as tf
import os, json, pickle

from transformer import Transformer
from utils.translate import Translate

# We do not want tensorflow loggings!
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
np.set_printoptions(suppress=True)

# Making tensorflow not to allocate the whole GPU
physical_devices = tf.config.list_physical_devices('GPU') 
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)

# loading the config file
with open('config.json') as f:
    config = json.load(f)

# path to save the tokenizers
source_tokenizer_save_path = config["save_paths"]["source_tokenizers_save_path"] + ".pickle"
target_tokenizer_save_path = config["save_paths"]["target_tokenizers_save_path"] + ".pickle"


with open(source_tokenizer_save_path, 'rb') as source_handle:
    source_tokenizer = pickle.load(source_handle)
    
with open(target_tokenizer_save_path, 'rb') as target_handle:
    target_tokenizer = pickle.load(target_handle)

# computing the vocab sizes
source_vocab_size, target_vocab_size = len(source_tokenizer.word_index) + 2, len(target_tokenizer.word_index) + 2
source_start_token, source_end_token = len(source_tokenizer.word_index), len(source_tokenizer.word_index) + 1
target_start_token, target_end_token = len(target_tokenizer.word_index), len(target_tokenizer.word_index) + 1

model =Transformer(
    source_vocab_size = source_vocab_size,
    target_vocab_size = target_vocab_size,
    number_of_layers = config["model_architecture"]["number_of_layers"],
    embedding_dimension = config["model_architecture"]["token_embedding_dimension"],
    number_of_attention_heads = config["model_architecture"]["number_of_attention_heads"],
    ffn_units = config["model_architecture"]["ffn_units"],
    dropout_rate = config["model_architecture"]["dropout_rate"],
    dropout_training = False,  # during inference dropout_training = False
    layernorm_epsilon = config["training_settings"]["layernorm_epsilon_value"])
model.load_weights(config["save_paths"]["model_save_path"])
print("Trained model loaded...", end = "\n")

while True:
    input_query_text = input("Enter the text: ")
    predicted_translation, attention_dict = Translate(model,
                                                    input_query_text,
                                                    source_tokenizer,
                                                    target_tokenizer,
                                                    config,
                                                    plot_attention_weights = True)
    print(f"Predicted Translation:", predicted_translation) 
    print(os.linesep)