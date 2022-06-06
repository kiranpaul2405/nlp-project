# Sentiment Analysis and Summarization using BERT and T5

## Project Description

The project aims to perform the following tasks:
   1.Identify the overall sentiment of a given text passage
   2.Identify the sentence wise sentiment of all sentences in a given text passage
   3.Generate a title for the passage

## Sentiment Analysis with BERT model

BERT(Bidirectional Encoder Representations from Transformers) is a transformer network developed by Google in 2018 for performing several NLP tasks. It is a neural network based model which is pretrained on two different tasks-Masked Language Modelling(Masking tokens and training bert to predict it) and Next Sentence Prediction(training bert to predict if a chosen sentence would follow a given one). Once fine tuned for a specific task bert is able to perform a variety of NLP tasks such as Sentiment Classification.

The bert model has two variation- bert base and bert large which differ by number of encoder layers. In this project bert base variation is used for sentiment prediction. A given text is first encoded into a set of token having a set of corresponding ids in a word corpus. Bert provides its own tokenizer for performing this operation, simplifying this task. Once encoded the tokens along with the associated labels can be fed into the bert model. The model learns the task at hand by looking at the encoded vectors and the sentiment. Training process can be done for any number of epoches.

In this project, the pretrained bert base model is then fine-tuned using a commonly used dataset for this particular problem- the IMDB dataset. This dataset is readily available from the tensorflow datasets package and can be used easily by loading into a dataframe. It consist of 25000 samples for training and a further 25000 samples for testing. Here the bert model is trained with 10000 samples.

The bert implementation used here is TFBertForSequenceClassification along with BertTokenizer from the Huggingface Transformers package.

To make use of GPU, training was done in Google Colab. Once training is completed the model is saved. Later it is downloaded into local machine. The saved model is later imported into a flask backend web application which classifies sentiment of text recieved from requests as positive or negative.

## Summarization with T5 model
   
