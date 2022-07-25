#!/usr/bin/env python3

###############################################################
# Method: Get
# EndPoint: /, /getModel and getInfo, if the endpoint is not valid, it will fallback to getInfo
# Query format and corresponding response format:: 
#   /?model=<model_name>&text=<text> -> list [E, M, E, E, M, M ...], E: English, M: Māori
#   /getModel -> list [model_name, model_name, model_name, ...]
#   /getInfo -> string The usage and the purpose of the API
###############################################################

from flask import Flask, redirect, request
from flask_restful import Api
import re
import pickle
import tensorflow as tf
from keras.utils.data_utils import pad_sequences
import numpy as np
import torch
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification

app = Flask(__name__)
api = Api(app)

# Parameters #
full_size_bilstm_model_path = 'models/bilstm.h5'
full_size_bilstm_tokenizer_path = 'models/tokenizerBilstm.pickle'
full_size_bilstm_lower_model_path = 'models/bilstmLower.h5'
full_size_bilstm_lower_tokenizer_path = 'models/tokenizerBilstmLower.pickle'

size_2_bilstm_model_path = 'models/bilstmSize2.h5'
size_2_bilstm_tokenizer_path = 'models/tokenizerBilstmSize2.pickle'
size_2_bilstm_lower_model_path = 'models/bilstmSize2Lower.h5'
size_2_bilstm_lower_tokenizer_path = 'models/tokenizerBilstmSize2Lower.pickle'

size_3_bilstm_model_path = 'models/bilstmSize3.h5'
size_3_bilstm_tokenizer_path = 'models/tokenizerBilstmSize3.pickle'
size_3_bilstm_lower_model_path = 'models/bilstmSize3Lower.h5'
size_3_bilstm_lower_tokenizer_path = 'models/tokenizerBilstmSize3Lower.pickle'

full_size_mbert_model_path = 'models/mbert'
full_size_mbert_lower_model_path = 'models/mbertLower'

mbert_tokenizer_path = 'models/tokenizerMbert'
# End of parameters #

# Load models #
with tf.device('/cpu:0'):
    full_size_bilstm_model = tf.keras.models.load_model(full_size_bilstm_model_path)
    full_size_bilstm_lower_model = tf.keras.models.load_model(full_size_bilstm_lower_model_path)

    size_2_bilstm_model = tf.keras.models.load_model(size_2_bilstm_model_path)
    size_2_bilstm_lower_model = tf.keras.models.load_model(size_2_bilstm_lower_model_path)

    size_3_bilstm_model = tf.keras.models.load_model(size_3_bilstm_model_path)
    size_3_bilstm_lower_model = tf.keras.models.load_model(size_3_bilstm_lower_model_path)

with open(full_size_bilstm_tokenizer_path, 'rb') as handle:
    full_size_bilstm_tokenizer = pickle.load(handle)
with open(full_size_bilstm_lower_tokenizer_path, 'rb') as handle:
    full_size_bilstm_lower_tokenizer = pickle.load(handle)

with open(size_2_bilstm_tokenizer_path, 'rb') as handle:
    size_2_bilstm_tokenizer = pickle.load(handle)
with open(size_2_bilstm_lower_tokenizer_path, 'rb') as handle:
    size_2_bilstm_lower_tokenizer = pickle.load(handle)

with open(size_3_bilstm_tokenizer_path, 'rb') as handle:
    size_3_bilstm_tokenizer = pickle.load(handle)
with open(size_3_bilstm_lower_tokenizer_path, 'rb') as handle:
    size_3_bilstm_lower_tokenizer = pickle.load(handle)

full_size_mbert_model = AutoModelForSequenceClassification.from_pretrained(full_size_mbert_model_path, num_labels=3)
for param in full_size_mbert_model.parameters():
    param.requires_grad_(False)
full_size_mbert_lower_model = AutoModelForSequenceClassification.from_pretrained(full_size_mbert_lower_model_path, num_labels=3)
for param in full_size_mbert_lower_model.parameters():
    param.requires_grad_(False)

mbert_tokenizer = AutoTokenizer.from_pretrained(mbert_tokenizer_path)
# End of load models #

# Functions #
## Clean the text
def cleanText(text):
    text = text.replace("“", '"').replace(
        "”", '"').replace("‘", "'").replace("’", "'")
    # replace non ascii char but keep the maori chars
    text = re.sub(r'[^\x00-\x7FāēīōūĀĒĪŌŪ]+', '', text)
    text = text.replace('\r', '  ').replace(
        '\n', '  ').replace('\t', '  ')  # remove \r \n \t
    text = text.replace(':', ': ').replace(';', '; ').replace(
        ',', ', ').replace('.', '. ')  # add space after the symbols
    while '  ' in text:
        text = text.replace('  ',  ' ')  # remove redundant spaces
    text = text.replace(' :', ':').replace(' ;', ';').replace(
        ' ,', ',').replace(' .', '.')  # remove space before the symbols
    # handle a.m and p.m
    text = text.replace('a. m', 'a.m').replace('p. m', 'p.m')
    return text.strip()

## BiLSTM model ##
### Detect the code switching point in a dynamic window
def sentenceCategory(sentence, padding_length, tokenizer, loaded_model):
    seq = tokenizer.texts_to_sequences([sentence])
    padded = pad_sequences(seq, maxlen=padding_length)
    predict = loaded_model.predict(padded) 
    classw = np.argmax(predict,axis=1)
    return int(classw[0])

def detectCodeSwitchingPointDynamicWindowVersion(x, w, tokenizer, loaded_model):
    p = w
    words_list = x.split()
    end = len(words_list)
    if w >= end and end > 2:
        w = end - 1
    elif end == 1:
        w = 1
    elif end == 2:
        w = 2
    else:
        pass

    if end < 1:
        return []

    elif end < 2:
        if re.search(u'[āēīōūĀĒĪŌŪ]', x):
            return [1]
        elif re.search(u'[bBcCdDfFgGjJlLqQsSvVxXyYzZ]', x):
            return [2]
        else:
            return [sentenceCategory(x, p, tokenizer, loaded_model)]

    elif end == 2:
        if not re.search(u'[āēīōūĀĒĪŌŪ]', x):
            tmp_result = sentenceCategory(x, p, tokenizer, loaded_model)
            if tmp_result == 1 and not re.search(u'[bBcCdDfFgGjJlLqQsSvVxXyYzZ]', x):
                return [1, 1]
            elif tmp_result == 2:
                return [2, 2]
            else:
                if sentenceCategory(words_list[0], p, tokenizer, loaded_model) == 1 and not re.search(u'[bBcCdDfFgGjJlLqQsSvVxXyYzZ]', words_list[0]):
                    return [1, 2]
                else:
                    return [2, 1]
        else:
            tmp_char_0 = re.search(u'[āēīōūĀĒĪŌŪ]', words_list[0])
            tmp_char_1 = re.search(u'[āēīōūĀĒĪŌŪ]', words_list[1])
            if tmp_char_0 and tmp_char_1:
                return [1, 1]
            if tmp_char_0 and not tmp_char_1:
                return [1, 2]
            else:
                return [2, 1]
    
    else:
        result = []
        ptr = 0
        while ptr < end:
            this_window = words_list[ptr:ptr+w]
            if ptr + w > end:
                w = end - ptr
            else:
                pass

            tmp_result = sentenceCategory(" ".join(this_window), p, tokenizer, loaded_model)
            if tmp_result == 1 and not re.search(u'[bBcCdDfFgGjJlLqQsSvVxXyYzZ]', " ".join(this_window)):
                result.extend([1 for _ in range(w)])
            elif tmp_result == 2 and not re.search(u'[āēīōūĀĒĪŌŪ]', " ".join(this_window)):
                result += [2 for _ in range(w)]
            else:
                if w >= 4 and w % 2 == 0:
                    result += detectCodeSwitchingPointDynamicWindowVersion(" ".join(this_window), w-2, tokenizer, loaded_model)
                else:
                    result += detectCodeSwitchingPointDynamicWindowVersion(" ".join(this_window), w-1, tokenizer, loaded_model)
            ptr += w
        return result
## End of BiLSTM model ##

## MBERT model ##
@torch.no_grad()
def sentenceCategoryMbertVersion(text: str, model) -> int:
    tokenized_text = mbert_tokenizer(text, padding="longest", truncation=True, return_tensors='pt')
    prediction = model(input_ids=tokenized_text["input_ids"], attention_mask=tokenized_text["attention_mask"], token_type_ids=tokenized_text["token_type_ids"])
    return prediction.logits.detach().cpu().numpy().argmax()

def detectCodeSwitchingPointMbertVersion(x: str, w: int, model) -> list():
    words_list = x.split()
    end = len(words_list)
    if w >= end and end > 2:
        w = end - 1
    elif end == 1:
        w = 1
    elif end == 2:
        w = 2
    else:
        pass

    if end < 1:
        return []

    elif end == 1:
        if re.search(u'[āēīōūĀĒĪŌŪ]', x):
            return [1]
        elif re.search(u'[bBcCdDfFgGjJlLqQsSvVxXyYzZ]', x):
            return [2]
        else:
            return [sentenceCategoryMbertVersion(x, model)]

    elif end == 2:
        if not re.search(u'[āēīōūĀĒĪŌŪ]', x):
            tmp_result = sentenceCategoryMbertVersion(x, model)
            if tmp_result == 1 and not re.search(u'[bBcCdDfFgGjJlLqQsSvVxXyYzZ]', x):
                return [1, 1]
            elif tmp_result == 2:
                return [2, 2]
            else:
                if sentenceCategoryMbertVersion(words_list[0], model) == 1 and not re.search(u'[bBcCdDfFgGjJlLqQsSvVxXyYzZ]', words_list[0]):
                    return [1, 2]
                else:
                    return [2, 1]
        else:
            if re.search(u'[āēīōūĀĒĪŌŪ]', words_list[0]) and re.search(u'[āēīōūĀĒĪŌŪ]', words_list[1]):
                return [1, 1]
            if re.search(u'[āēīōūĀĒĪŌŪ]', words_list[0]) and not re.search(u'[āēīōūĀĒĪŌŪ]', words_list[1]):
                return [1, 2]
            else:
                return [2, 1]
    
    else:
        result = []
        ptr = 0
        while ptr < end:
            this_window = words_list[ptr:ptr+w]
            if ptr + w > end:
                w = end - ptr
            else:
                pass
            if sentenceCategoryMbertVersion(" ".join(this_window), model) == 1 and not re.search(u'[bBcCdDfFgGjJlLqQsSvVxXyYzZ]', " ".join(this_window)):
                result.extend([1 for _ in range(w)])
            elif sentenceCategoryMbertVersion(" ".join(this_window), model) == 2 and not re.search(u'[āēīōūĀĒĪŌŪ]', " ".join(this_window)):
                result += [2 for _ in range(w)]
            else:
                if w >= 4 and w % 2 == 0:
                    result += detectCodeSwitchingPointMbertVersion(" ".join(this_window), w-2, model)
                else:
                    result += detectCodeSwitchingPointMbertVersion(" ".join(this_window), w-1, model)
            ptr += w
        return result
## End of Mbert model ##

def transfrom(a: list) -> list:
    for index, item in enumerate(a):
        if item == 1:
            a[index] = 'M'
        elif item == 2:
            a[index] = 'E'
        else:
            a[index] = 'U'
    a = ", ".join(a)
    return a
# End of functions #

@app.route('/favicon.ico', methods=['GET'])
def favicon():
    return redirect("https://aotearoavoices.nz/favicon.ico")

@app.route('/getModel', methods=['GET'])
def getModel():
        return '<title>M/E CW Detection API</title>Avaliable Models are: size_2_bilstm, size_2_bilstm_lower, size_3_bilstm, size_3_bilstm_lower, full_size_bilstm, full_size_bilstm_lower, full_size_mbert, full_size_mbert_lower.<br/><br/>For more information, please visit <a href="./getInfo">getInfo</a>.<div style="position:fixed;bottom:0;background-color:white;width:100%"><div style="text-align:center"><label style="padding-right:4px;">Copyright © 2022</label><a href="https://speechresearch.auckland.ac.nz/">Speech Research Group @ UoA</a><label>. All rights reserved.</label></div></div>'

@app.route('/getInfo', methods=['GET'])
def getInfo():
    return '<title>M/E CW Detection API</title>This API is used to detect the code switching point between English and Māori.<br/><br/>Query format: /?model=&ltmodel_name&gt&text=&lttext&gt<br/><br/>Response format: E, M, E, E, M, M ..., where E is English, M is Māori<br/><br/>To get the list of models, visit <a href="./getModel">getModel</a>.<br/><br/>To get the usage and the purpose of the API, visit <a href="./getInfo">getInfo</a>.<r/><br/><br/>For more information of the definition of code switch detection and the models used, please visit: <a href="https://openreview.net/forum?id=rAxl_GibSWq">https://openreview.net/forum?id=rAxl_GibSWq</a><div style="position:fixed;bottom:0;background-color:white;width:100%"><div style="text-align:center"><label style="padding-right:4px;">Copyright © 2022</label><a href="https://speechresearch.auckland.ac.nz/">Speech Research Group @ UoA</a><label>. All rights reserved.</label></div></div>'

@app.errorhandler(404)
def not_found(error):
    return redirect('/getInfo') # Modify

@app.route('/', methods=['GET'])
def detect():
    args = request.args
    if 'model' not in args or 'text' not in args:
        return redirect('/getInfo') # Modify
    else:
        model_name = args['model']
        text = args['text']
        if model_name == "size_2_bilstm":
            return transfrom(detectCodeSwitchingPointDynamicWindowVersion(text, 2, size_2_bilstm_tokenizer, size_2_bilstm_model))
        elif model_name == "size_2_bilstm_lower":
            return transfrom(detectCodeSwitchingPointDynamicWindowVersion(text.lower(), 2, size_2_bilstm_lower_tokenizer, size_2_bilstm_lower_model))

        elif model_name == "size_3_bilstm":
            return transfrom(detectCodeSwitchingPointDynamicWindowVersion(text, 3, size_3_bilstm_tokenizer, size_3_bilstm_model))
        elif model_name == "size_3_bilstm_lower":
            return transfrom(detectCodeSwitchingPointDynamicWindowVersion(text.lower(), 3, size_3_bilstm_lower_tokenizer, size_3_bilstm_lower_model))
        
        elif model_name == "full_size_bilstm":
            return transfrom(detectCodeSwitchingPointDynamicWindowVersion(text, 250, full_size_bilstm_tokenizer, full_size_bilstm_model))
        elif model_name == "full_size_bilstm_lower":
            return transfrom(detectCodeSwitchingPointDynamicWindowVersion(text.lower(), 250, full_size_bilstm_lower_tokenizer, full_size_bilstm_lower_model))
        
        elif model_name == "full_size_mbert":
            return transfrom(detectCodeSwitchingPointMbertVersion(text, 4, full_size_mbert_model))
        elif model_name == "full_size_mbert_lower":
            return transfrom(detectCodeSwitchingPointMbertVersion(text.lower(), 4, full_size_mbert_lower_model))
        
        else:
            return redirect('/getInfo') # Modify

if __name__ == '__main__':
    # Bind 127.0.0.1:8500, only accept connections from localhost
    app.run(host='127.0.0.1', port=8500, debug=True)
    