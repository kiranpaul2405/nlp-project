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

To make use of GPU, training was done in Google Colab. Once training is completed the model is saved. Later it is downloaded into local machine. The saved model is later imported into a flask backend web application which classifies sentiment of text recieved from requests as positive or negative.The backend will produce the overall sentiment as well as the sentence wise sentiment for each request text. 

## Summarization with T5 model
   
T5 is another transformer model used for various NLP tasks such as Question Answering, text generation as well as Abstractive text summarization. Here we use its abstractive text summarization capability for generating title for a given text. Different variations are available for this model: T5-small,T5-base, T5-large etc.
Here T5-small has been used.

The model implementation used here is T5ForConditionalGeneration. Its a pretrained model which comes with comes with with its own tokenizer- T5Tokenizer from the transformers package. The pretrained model can be simply used as such for summarization.

In this project, this model is loaded in the same backend used for the sentiment analysis directly from the transformers package. Inputs are encoded using the provided tokenizer and fed into the network.It will generate tokens which can be decoded to get the summary. This is performed on the same text sent to the backend for sentiment analysis.

## Requirements 
Python
transformers
tensorflow
flask
flask_cors
tensorflow_datasets

## Resources and how to run the project

This project repo consist of two files:

### Sentiment_Analyzer.ipynb

This file contains the code for trainig the Bert model for sentiment analysis. To run it, download it from this github repo and run it in google colab or jupyter notebook. While saving the model provide a path of your convenience. If training is done using Google Colab, download the model and also the generated config file.

### app.py

This file contains the code for flask backend application. Before running the backend all specified dependencies must be installed in the environment. The path in which the model is saved must be modified as per the requirement. To run the application go to the folder in which app.py is located from a command prompt and 
type python app.py. This will start the backend at 127.0.0.1:5000. Sample api request and response is given in appendix. 

## Sample API Request and response formats

The API is a POST API
API: http://127.0.0.1:5000/predict   

Request body:{"review":"This is good.This is bad"}

Response body:{
    "results": {
        "sentence_wise_sentiment": [
            {
                "sentence": "This is good.",
                "sentiment": "positive"
            },
            {
                "sentence": "This is bad",
                "sentiment": "negative"
            }
        ],
        "sentiment": "positive",
        "title": "It is good"
    }
}

