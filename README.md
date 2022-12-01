## Attention Is All You Need Implementation (TensorFlow 2.x) 🚀

This repository contains the TensorFlow implementation of the paper (:link: [Attention Is All You Need](https://arxiv.org/abs/1706.03762)). Transformers are the new SOTA, not only in Natural Language Processing but also in Vision.This implementation can be used to perform any sequence to sequence task with some minimal code changes. 

<div align="center">
    <img src="images\transformer_picture.png" width="400" height = "350">
</div>

### Setup
Tensorflow GPU is necessary for fast training and inference. All the dependencies can be installed using the following
commmand.

```
git clone https://github.com/TheTensorDude/Transformer-TF2.x.git
cd Transformer-TF2.x
pip install -r requirements.txt
```

### Training the Transformer :metal:
Before training the hyperparameters need to be set in the config.json file.

```json 
{
    "model_architecture" : {
        "token_embedding_dimension" : 128,
        "number_of_attention_heads" : 8,
        "number_of_layers" : 2,
        "ffn_units" : 512,
        "max_sentence_length" : 70,
        "dropout_rate" : 0.3
    },
    "training_settings" : {
        "batch_size" : 128,
        "epochs" : 70,
        "test_size" : 0.2,
        "warmup_steps" : 4000,
        "generate_training_stats" : true,
        "plot_training_stats" : false,
        "training_stats_save_dir" : "runs",
        "layernorm_epsilon_value" : 1e-6,
        "dropout_training" : true  
    },
    "data" : {
        "data_path" : "data//Translation//spa.txt",
        "approximate_source_tokens" : null,
        "approximate_target_tokens" : null
    },
    "save_paths" : {
        "model_save_path" : "model/english-to-spanish-ckpt",
        "source_tokenizers_save_path" : "model//english_tokenizer",
        "target_tokenizers_save_path" : "model//spanish_tokenizer"
    },
    "inference_settings" : {
        "visualise_attention_layer" : "decoder_layer_cross_attention_2",
        "attention_plot_save_dir" : "runs/exp"
    }
}
```

After setting up the values and the paths the next step is to run the below command and the training will be started. With respect to the csv file, the model will start training on the english to spanish translation task. The CSV file should have two columns, source and target.
```
python train_transformer.py
```

Output 

```

Epoch: 46, Training Loss: 38.76237487792969, Training Accuracy 0.8394668698310852, Test Loss : 33.60411834716797, Test Accuracy : 0.6773480772972107
Model saved...
Source text Input: If you do such a foolish thing, people will laugh at you.
Predicted Translation: si haces una tontería es una persona muy molesto contigo
Original Translation: Si haces algo así de estúpido, se burlarán de ti.


Batch: 1, Batch Loss: 0.05608042702078819, Batch Accuracy: 0.8525499105453491
Batch: 2, Batch Loss: 0.06947806477546692, Batch Accuracy: 0.8204845786094666
Batch: 3, Batch Loss: 0.06688302010297775, Batch Accuracy: 0.8291404843330383
Batch: 4, Batch Loss: 0.05281899496912956, Batch Accuracy: 0.8342857360839844
Batch: 5, Batch Loss: 0.05348208174109459, Batch Accuracy: 0.8553459048271179
```

### Custom Learning rate

The original transformer was trained using a custom learning rate scheduler. The scheduler is implemented in utils/utils.py.
```python

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

```
Below is the plot of the learning rate and the training steps.

<img src="runs\learning_rate.png" width="300" height = "200">


### Inference and Attention maps
After the model is trained, inference can be generated using the following command.
```
python generate_inference.py
```
Understand that source language is English and target translation is spanish.

```
Trained model loaded...
Enter the text: I love you
Predicted Translation: te amo


Enter the text: I love my country
Predicted Translation: amo a mi país
```

Below are the attention maps for the spanish translation of the english sentence "i love you".

<img src="images\attention_head_8.jpg" width="400" height = "300">

### Notes
- If the value of approximate_source_tokens is less than the actual number of token then the loss becomes nan.
- The dropout_training key should be false while generating inference.
