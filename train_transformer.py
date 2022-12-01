import numpy as np
import pandas as pd
import tensorflow as tf
from random import randint
import os, json, pickle, re
from pprint import pprint
import matplotlib.pyplot as plt

from trainer import  trainer
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from utils.loss import MaskedCCEloss
from transformer import Transformer
from utils.translate import Translate
from utils.performance_metrics import masked_accuracy
from utils.utils import get_transformer_logits, get_input_ids_from_texts, plot_loss, plot_accuracy, plot_learning_rate, CustomLRForAdam

# We do not want tensorflow loggings!
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

# Making tensorflow not to allocate the whole GPU
physical_devices = tf.config.list_physical_devices('GPU') 
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)

# loading the config file
with open('config.json') as f:
    config = json.load(f)

print("Given config...")
pprint(config)

# loading the dataframe
df = pd.read_csv(config["data"]["data_path"], sep = "\t")
# -------------------- Preprocessing ---------------------
df.dropna(inplace = True)
# dropping duplicates and resetting the indices
df.drop_duplicates(subset = df.columns[0], keep="first", inplace=True)
df.reset_index(drop=True, inplace = True)
# renaming the columns
df.columns = ["source", "target"]
# shuffling the dataframe
df = df.sample(frac=1).reset_index(drop=True)


def clean_english_phrase(phrase):
    phrase = re.sub(r"won't", "will not", phrase)
    phrase = re.sub(r"dont", "do not", phrase)
    phrase = re.sub(r"prolly", "probably", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase

# cleaning the source sentences
df["source"] = df.source.apply(lambda x: clean_english_phrase(x))

# Initializing the whitespace tokenizers using keras, lower = True means it will lower the tokens
source_tokenizer = Tokenizer(num_words= config["data"]["approximate_source_tokens"], 
                             lower = True)
target_tokenizer = Tokenizer(num_words= config["data"]["approximate_target_tokens"], 
                             lower = True)

# fitting the tokenizers
source_tokenizer.fit_on_texts(df.source)
target_tokenizer.fit_on_texts(df.target)

# (start token + end token)[2] + all_words 
source_vocab_size, target_vocab_size = len(source_tokenizer.word_index) + 2, len(target_tokenizer.word_index) + 2

# path to save the tokenizers
source_tokenizer_save_path = config["save_paths"]["source_tokenizers_save_path"] + ".pickle"
target_tokenizer_save_path = config["save_paths"]["target_tokenizers_save_path"] + ".pickle"

# saving the tokenizers
with open(source_tokenizer_save_path, 'wb') as source_handle:
    pickle.dump(source_tokenizer, source_handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(target_tokenizer_save_path, 'wb') as target_handle:
    pickle.dump(target_tokenizer, target_handle, protocol=pickle.HIGHEST_PROTOCOL)
print(f"Tokenizers are saved...")

# getting input ids from raw texts 
source_input_ids = get_input_ids_from_texts(texts = df.source, tokenizer = source_tokenizer)
target_input_ids = get_input_ids_from_texts(texts = df.target, tokenizer = target_tokenizer)
labels = get_input_ids_from_texts(texts = df.target, tokenizer = target_tokenizer, labels = True)


# padding the ragged tensor data so that the input to the model becomes rectangular
source_input_ids = pad_sequences(source_input_ids,
                          maxlen=config["model_architecture"]["max_sentence_length"],
                          padding="post")

target_input_ids = pad_sequences(target_input_ids, 
                          maxlen=config["model_architecture"]["max_sentence_length"],
                          padding="post")

labels = pad_sequences(labels,
                      maxlen=config["model_architecture"]["max_sentence_length"],
                      padding="post")


# splitting the data into training and testing
train_source_input_ids, test_source_input_ids = train_test_split(source_input_ids, 
                                                                 shuffle = False,
                                                                test_size=config["training_settings"]["test_size"])
train_target_input_ids, test_target_input_ids = train_test_split(target_input_ids,
                                                                 shuffle = False,
                                                                test_size=config["training_settings"]["test_size"])
train_labels, test_labels = train_test_split(labels, 
                                            shuffle = False,
                                            test_size=config["training_settings"]["test_size"])

# building tensor slices from numpy array of data
# for training
train_source_english_sentence = tf.data.Dataset.from_tensor_slices(train_source_input_ids)
train_target_bengali_sentence = tf.data.Dataset.from_tensor_slices(train_target_input_ids)
train_labels = tf.data.Dataset.from_tensor_slices(train_labels)
# for test
test_source_english_sentence = tf.data.Dataset.from_tensor_slices(test_source_input_ids)
test_target_bengali_sentence = tf.data.Dataset.from_tensor_slices(test_target_input_ids)
test_labels = tf.data.Dataset.from_tensor_slices(test_labels)

# zipping all the input variables and output variables into tf.data.Datasets object
train_dataset = tf.data.Dataset.zip((train_source_english_sentence, train_target_bengali_sentence, train_labels))\
    .cache()\
    .batch(batch_size=config["training_settings"]["batch_size"])\
    .prefetch(tf.data.experimental.AUTOTUNE)
    
test_dataset = tf.data.Dataset.zip((test_source_english_sentence, test_target_bengali_sentence, test_labels))\
    .cache()\
    .batch(batch_size=config["training_settings"]["batch_size"])\
    .prefetch(tf.data.experimental.AUTOTUNE)
    
    
    
print("Building the model...")
model = Transformer(
    source_vocab_size = source_vocab_size,
    target_vocab_size = target_vocab_size,
    number_of_layers = config["model_architecture"]["number_of_layers"],
    embedding_dimension = config["model_architecture"]["token_embedding_dimension"],
    number_of_attention_heads = config["model_architecture"]["number_of_attention_heads"],
    ffn_units = config["model_architecture"]["ffn_units"],
    dropout_rate = config["model_architecture"]["dropout_rate"],
    dropout_training = config["training_settings"]["dropout_training"],
    layernorm_epsilon = config["training_settings"]["layernorm_epsilon_value"]
    )



print("Model is ready to train...")

custom_learning_rate = CustomLRForAdam(
                embedding_dimension = config["model_architecture"]["token_embedding_dimension"],
                warmup_steps=config["training_settings"]["warmup_steps"])

optim = tf.optimizers.Adam(learning_rate=custom_learning_rate)

if config["training_settings"]["generate_training_stats"]:
    plot_learning_rate(custom_learning_rate(step = tf.range(50000, dtype=tf.float32)),
                    config["training_settings"]["training_stats_save_dir"],
                    config["training_settings"]["plot_training_stats"])

print("starting the training...")
print(os.linesep)      

training_info = {
    "Training Loss" : [],
    "Training Accuracy" : [],
    "Test Loss" : [],
    "Test Accuracy" : []
}

max_accuracy = 0

for epoch in range(config["training_settings"]["epochs"]):
    epochInfo = {
      "epoch_training_loss" : [],
      "epoch_training_accuracy" : [],
      "epoch_test_loss" : [],
      "epoch_test_accuracy" : []
    }
    
    for num_training_batch, (target_source_tensor, target_target_tensor, target_label) in enumerate(train_dataset):
        loss, accuracy = trainer(model,
                                target_source_tensor, 
                                target_target_tensor, 
                                target_label,
                                optim)
        print(f"Batch: {num_training_batch + 1}, Batch Loss: {loss.numpy()}, Batch Accuracy: {accuracy.numpy()}")
        epochInfo["epoch_training_loss"].append(loss.numpy())
        epochInfo["epoch_training_accuracy"].append(accuracy.numpy())
    
    print(os.linesep)    
    
    for test_source_tensor, test_target_tensor, test_labels in test_dataset:
        test_predicted_logits = get_transformer_logits(model,
                                                       test_source_tensor,
                                                       test_target_tensor)
        test_batch_accuracy = masked_accuracy(test_labels, test_predicted_logits)
        test_batch_loss = MaskedCCEloss(test_labels, test_predicted_logits)
        epochInfo["epoch_test_loss"].append(test_batch_loss.numpy())
        epochInfo["epoch_test_accuracy"].append(test_batch_accuracy.numpy())
    
    epoch_training_acc = np.median(epochInfo["epoch_training_accuracy"])
    epoch_test_acc = np.median(epochInfo["epoch_test_accuracy"])
    
    epoch_training_loss = np.sum(epochInfo["epoch_training_loss"]) 
    epoch_test_loss = np.sum(epochInfo["epoch_test_loss"]) 
    

        
    print(f"Epoch: {epoch + 1}, Training Loss: {epoch_training_loss}, Training Accuracy {epoch_training_acc}, Test Loss : {epoch_test_loss}, Test Accuracy : {epoch_test_acc}", end = "\n")
    
    training_info["Training Loss"].append(epoch_training_loss)
    training_info["Test Loss"].append(epoch_test_loss)
    
    training_info["Training Accuracy"].append(epoch_training_acc)
    training_info["Test Accuracy"].append(epoch_test_acc)
    
        
    if epoch_test_acc > max_accuracy:
        max_accuracy = epoch_test_acc
        model.save_weights(config["save_paths"]["model_save_path"])          
        print(f"Model saved...")
        
    chosen_index = randint(a = 0, b = df.shape[0])    
    chosen_source_text = df.source.iloc[chosen_index]
    predicted_translation, _ = Translate(model = model,
                    source_language = chosen_source_text,
                    source_tokenizer = source_tokenizer,
                    target_tokenizer = target_tokenizer,
                    config = config)
    print(f"Source text Input:" , chosen_source_text)
    print(f"Predicted Translation:", predicted_translation[0])
    print(f"Original Translation:",  df.target.iloc[chosen_index])
    print(os.linesep)        
    
if config["training_settings"]["generate_training_stats"]:
    plot_loss(training_info,
              config["training_settings"]["training_stats_save_dir"],
              config["training_settings"]["plot_training_stats"])
    
    plot_accuracy(training_info,
              config["training_settings"]["training_stats_save_dir"],
              config["training_settings"]["plot_training_stats"])
    



