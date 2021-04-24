# Text CLassifier

A simple text classifier to classify text into news category. The main objective of this project is to understand the fundamentals of NLP.

### Data Processing

The data is first converted to Ngrams and is now a tensor. In NLP, since the length of each sequence is variable, it is important to create batches carefully. In this model, since we are using an EmbeddingBag layer, we neet to pass text and their offsets. So each batch is a tuple of text and their offsets (and their labels during training). 

![image](https://user-images.githubusercontent.com/7227383/115950536-1b237100-a4dc-11eb-8bae-6a22330b0271.png)

### Model 

The embeddingBag layer computes the mean of the embedding for each of the sequence in the mini-batch, which is a vector of dimension ```EMBED_DIM``` for each sequence. This then goes through the linear fully-connected layer.

![image](https://user-images.githubusercontent.com/7227383/115950710-2f1ba280-a4dd-11eb-9e97-672cf0cdc9d9.png)

