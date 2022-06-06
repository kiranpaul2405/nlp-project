from flask import Flask,request,jsonify
from transformers import T5Tokenizer,T5ForConditionalGeneration
from transformers import TFBertForSequenceClassification,BertTokenizer
from flask_cors import CORS
import tensorflow as tf
app=Flask(__name__)
CORS(app)
@app.route('/predict',methods=['POST'])

def predict():
    
    review=request.form.get('review')
    result={}
    print('Request recieved....Loading model for summarization..... ')
    tokenizer=T5Tokenizer.from_pretrained('t5-small')
    summarizer_model=T5ForConditionalGeneration.from_pretrained('t5-small')
    print('Summarizer Model Loaded!!!!')
    preprocessed_text=review.strip().replace('\n','')
    preprocessed_text=preprocessed_text.replace('<br />','')
    preprocessed_text=preprocessed_text.replace('!','.')
    t5_input='summarize: '+preprocessed_text
    tokenized_text=tokenizer.encode(t5_input,return_tensors='pt',max_length=512)
    summary_ids=summarizer_model.generate(tokenized_text,min_length=5,max_length=10)
    title=tokenizer.decode(summary_ids[0],skip_special_tokens=True)
    result['title']=title
    print('Title is generated')
    
    print('loading fine tuned model for sentiment analysis...')
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    model=TFBertForSequenceClassification.from_pretrained(r"G:\flask-app\sentiment_analyser\\")
    print('model loading completed!!!')
    predict_input = bert_tokenizer.encode(preprocessed_text,truncation=True,padding=True,return_tensors="tf")
    tf_output = model.predict(predict_input)[0]
    tf_prediction = tf.nn.softmax(tf_output, axis=1)
    labels = ['negative','positive'] #(0:negative, 1:positive)
    label = tf.argmax(tf_prediction, axis=1)
    label = label.numpy()
    sentiment=labels[label[0]]
    result['sentiment']=sentiment
    print('Overall sentimence is predicted')
    
    sentence_wise_sentiment=[]
    sentence_list=preprocessed_text.split('.')
    empty=''
    sentence_list=[i for i in sentence_list if i!=empty]
    for text in sentence_list:
        sentence_dict={}
        sentence_dict['sentence']=text
        predict_input = bert_tokenizer.encode(preprocessed_text,truncation=True,padding=True,return_tensors="tf")
        tf_output = model.predict(predict_input)[0]
        tf_prediction = tf.nn.softmax(tf_output, axis=1)
        labels = ['negative','positive'] #(0:negative, 1:positive)
        label = tf.argmax(tf_prediction, axis=1)
        label = label.numpy()
        sentiment=labels[label[0]]
        sentence_dict['sentiment']=sentiment
        sentence_wise_sentiment.append(sentence_dict)
    result['sentence_wise_sentiment']=sentence_wise_sentiment
    print('Sentence wise sentiment is predicted')
    print('Request processing complete...Returning response')    
    return jsonify(results=result)


if __name__=='__main__':
    
    app.run(debug=False)