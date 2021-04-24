# Text CLassifier

A simple text classifier to classify text into news category. The main objective of this project is to understand the fundamentals of NLP.

### Data Processing

The data is first converted to Ngrams and is now a tensor. In NLP, since the length of each sequence is variable, it is important to create batches carefully. In this model, since we are using an EmbeddingBag layer, we neet to pass text and their offsets. So each batch is a tuple of text and their offsets (and their labels during training). 

![image](https://user-images.githubusercontent.com/7227383/115950536-1b237100-a4dc-11eb-8bae-6a22330b0271.png)

### Basic Model 

The EmbeddingBag layer computes the mean of the embedding for each of the sequence in the mini-batch, which is a vector of dimension ```EMBED_DIM``` for each sequence. This then goes through the linear fully-connected layer.

![image](https://user-images.githubusercontent.com/7227383/115950710-2f1ba280-a4dd-11eb-9e97-672cf0cdc9d9.png)


### LSTM Model

As a first try, add a single LSTM layer to the model. 

![image](https://user-images.githubusercontent.com/7227383/115951315-68094680-a4e0-11eb-9dc4-de621fde8863.png)

Since LSTM takes input of dimension `(seg_len, batch_size, embed_dim)`, we pad each sequence with zeros until it is of size `seq_len`. We also need to use Embedding layer instead of EmbeddignBag, since we do not wish to sum over the sequence length.

![image](https://user-images.githubusercontent.com/7227383/115951620-f0d4b200-a4e1-11eb-8775-7fd5f756fae7.png)

The 3D tensor now goes through the LSTM layer. The LSTM can be rolled out in time, with the model sharing the same parameters across the time steps.
PyTorch outputs three tensors for LSTM layer:
1. the tensor that stacks all outputs from `h[0]` to `h[n]` (dim: `seq_len x batch_size x hidden_dim_of_lstm`)
2. the last output `h[n]` of dimension `1 x batch_size x hidden_dim_of_lstm`
3. the last cell state

Since we are interested in classifying the entire sequence as one category, we pass the last output to a linear layer. This then gives a score for each class.

![image](https://user-images.githubusercontent.com/7227383/115952423-1794e780-a4e6-11eb-8251-64296c6a6319.png)
