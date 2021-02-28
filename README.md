# StumbleUponChallenge
A classifier built for the StumbleUpon Kaggle Challenge 
-made by Sankalp Agrawal

---Modules used- Pytorch, torchtext, numpy, sklearn, nltk 
---Pretrained word embeddings used for constructing features (GloVe embeddings) 
---Heterogenous input has been considered (both numerical and text)
---Kindly refer the code for detailed documentation

The train.tsv file was split 80:20 into train and validation sets. Sizes obtained are-
-Size of train dataset:  5916
-Size of validation dataset:  1479
Following is the approach taken to address this task-

## Features considered

### Numerical features
The columns with numerical values (Column #5 to the last column) are taken into account. They are initially present as a string after being parsed from the csv files. All values are in a list inside a string, so it is preprocessed to get float values in a proper list. This list is passed to numerical_field, the field which wilil be used for making a tabular dataset and combining different types of features together. (NaN values denoted by ? are converted to 0)

### Text features
The boilerplate text field is used for this. For data cleaning, "title" word is removed from each text sample.
```
Preprocessing steps done on text-
1. Stopword Removal
2. Tokenization (using Regexptokeniser)
3. Lemmatization (using WordNetLemmatizer)
4. Converting words to lower case
```
This preprocessed text is passed to the content_field. The Glove embeddings are used her to build the vocabalury, of size 100 dimension. The vector obtained from these embeddings undergoes PCA as a postprocessing step (Since, the length of each vector is variable and very large in size). PCA reduces the dimensionality and number of features are effectively reduced to 5 (since batch_size is taken to be 5).
This content_field, along with label_field (obtained from integer labels) and numerical_field are used to make a TabularDataset, which in turn is used for obtaining the BucketIterator. Batch_size=5 is specified for the iterator.


## Model used
A simple RNN classsifier is made for this task. 
```
Model hyperparameters-
1. Learning rate: 
2. Tokenization (using Regexptokeniser)
3. Lemmatization (using WordNetLemmatizer)
4. Converting words to lower case
```

## Training and Evaluation

## References
For the model-
https://www.kaggle.com/kuldeep7688/simple-rnn-using-glove-embeddings-in-pytorch
Help in adding features, errors-
https://stackoverflow.com/questions/54267919/how-can-i-add-a-feature-using-torchtext
https://discuss.pytorch.org/t/cuda-error-runtimeerror-cudnn-status-execution-failed/17625
https://discuss.pytorch.org/t/runtimeerror-cuda-error-cublas-status-alloc-failed-when-calling-cublascreate-handle/78545/5
