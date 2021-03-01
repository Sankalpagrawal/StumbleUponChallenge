# StumbleUponChallenge
A classifier built for the StumbleUpon Kaggle Challenge <br/>
-made by Sankalp Agrawal

---Modules used- Pytorch, torchtext, numpy, sklearn, nltk <br/>
---Pretrained word embeddings used for constructing features (GloVe embeddings) <br/>
---Heterogenous input has been considered (both numerical and text) <br/>
---Kindly refer the code for detailed documentation <br/>

The train.tsv file was split 80:20 into train and validation sets. Sizes obtained are- <br/>
-Size of train dataset:  5916 <br/>
-Size of validation dataset:  1479 <br/>
These are written temporarily to trainset.csv, valset.csv and testset.csv files present in the repo. <br/>
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
This preprocessed text is passed to the content_field. The Glove embeddings are used her to build the vocabalury, of size 100 dimension. The vector obtained from these embeddings undergoes PCA as a postprocessing step (Since, the length of each vector is variable and very large in size). PCA reduces the dimensionality and number of features are effectively reduced to 5. <br/>
This content_field, along with label_field (obtained from integer labels) and numerical_field are used to make a TabularDataset, which in turn is used for obtaining the BucketIterator. Batch_size=5 is specified for the iterator.


## Model used

A simple RNN classsifier is made for this task. Embedding layer is used first (for learning the word embeddings), followed by an RNN unit, which is useful for modeling sequential data. The output is transmitted to a fully connected layer to get the final output scores for the 2 classes.

```
Model hyperparameters-
1. Learning rate: 1e-5
2. Optimizer used:  Adam
3. Loss function: Binary Cross Entropy Loss 
4. Hidden Layer dimension: 64
5. Dropout layer used with p=0.4
6. Epochs used for training: 50
```

## Training and Evaluation

### Training process

The batches are iterated over and loss function is minimized via Adam optimizer. The text+numerical features are concatenated together to be sent as input to the model. Since batch size=5, a 22x5(numerical)+5x5(text)=27x5 sized input feature vector is obtained. The values in this vector are normalized between 0 to 50, since values between [0,num_embeddings-1] are only accepted while training using Glove word embeddings. Training loss v/s epoch, and accuracy v/s epoch plots are shown in the ipynb notebook.

### Evaluation Results

Accuracy obtained on training set: **56.452%**
Accuracy obtianed on validation set: **56.385%**
ROC-AUC score obtained on the complete dataset: **0.5605** <br/>
Classwise precision,recall (obtained through classsification report): <br/>
**Class 0-** <br/>
Precision: 0.57
Recall: 0.43 <br/>
**Class 1-** <br/>
Precision: 0.56
Recall: 0.69

The urlids and predicted labels for the test set are stored in submission_file.csv.

## References
For the RNN model- <br/>
1. https://www.kaggle.com/kuldeep7688/simple-rnn-using-glove-embeddings-in-pytorch <br/>
For help in adding features, errors, these links were fruitful- <br/>
2. https://stackoverflow.com/questions/54267919/how-can-i-add-a-feature-using-torchtext
3. https://discuss.pytorch.org/t/cuda-error-runtimeerror-cudnn-status-execution-failed/17625
4. https://discuss.pytorch.org/t/runtimeerror-cuda-error-cublas-status-alloc-failed-when-calling-cublascreate-handle/78545/5 
5. Pytorch, sklearn documentations
