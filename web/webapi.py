#!/usr/bin/env python3

###############################################################
# Method: Get
# EndPoint: /, /getModel and getInfo, if the endpoint is not valid, it will fallback to getInfo
# Query format and corresponding response format:: 
#   /?model=<model_name>&text=<text> -> list [E, M, E, E, M, M ...], E: English, M: Māori
#   /getModel -> list [model_name, model_name, model_name, ...]
#   /getInfo -> string The usage and the purpose of the API
###############################################################

from flask import Flask, redirect, request, jsonify
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

size_2_mbert_model_path = 'models/mbertSize2'
size_2_mbert_lower_model_path = 'models/mbertSize2Lower'

size_3_mbert_model_path = 'models/mbertSize3'
size_3_mbert_lower_model_path = 'models/mbertSize3Lower'

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

size_2_mbert_model = AutoModelForSequenceClassification.from_pretrained(size_2_mbert_model_path, num_labels=3)
for param in size_2_mbert_model.parameters():
    param.requires_grad_(False)
size_2_mbert_lower_model = AutoModelForSequenceClassification.from_pretrained(size_2_mbert_lower_model_path, num_labels=3)
for param in size_2_mbert_lower_model.parameters():
    param.requires_grad_(False)

size_3_mbert_model = AutoModelForSequenceClassification.from_pretrained(size_3_mbert_model_path, num_labels=3)
for param in size_3_mbert_model.parameters():
    param.requires_grad_(False)
size_3_mbert_lower_model = AutoModelForSequenceClassification.from_pretrained(size_3_mbert_lower_model_path, num_labels=3)
for param in size_3_mbert_lower_model.parameters():
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

## Compare the list 
def listCmp(a:list, b:list) -> bool: # Todo: fix this after literature review
    return abs(a[1] - b[1]) <= 0.05 and abs(a[2] - b[2]) <= 0.05

## BiLSTM model ##
### Detect the code switching point in a dynamic window
def sentenceCategory(sentence, padding_length, tokenizer, loaded_model):
    seq = tokenizer.texts_to_sequences([sentence])
    padded = pad_sequences(seq, maxlen=padding_length)
    return list(loaded_model.predict(padded, verbose=0)[0])

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

    elif end == 1:
        if re.search(u'[āēīōūĀĒĪŌŪ]', x):
            return [[0, 1, 0]]
        elif re.search(u'[bBcCdDfFgGjJlLqQsSvVxXyYzZ]', x):
            return [[0, 0, 1]]
        else:
            return [sentenceCategory(x, p, tokenizer, loaded_model)]

    elif end == 2:
        tmp_result_list = sentenceCategory(x, p, tokenizer, loaded_model)
        tmp_result = np.argmax(tmp_result_list)
        if tmp_result == 1 and not re.search(u'[āēīōūĀĒĪŌŪ]', x) and not re.search(u'[bBcCdDfFgGjJlLqQsSvVxXyYzZ]', x):
            return [tmp_result_list, tmp_result_list]
        else:
            return detectCodeSwitchingPointDynamicWindowVersion(words_list[0], 1, tokenizer, loaded_model) + detectCodeSwitchingPointDynamicWindowVersion(words_list[1], 1, tokenizer, loaded_model)
    
    else:
        result = []
        ptr = 0
        while ptr < end:
            this_window = words_list[ptr:ptr+w]
            if ptr + w > end:
                w = end - ptr
            else:
                pass
            
            tmp_result_list = list(sentenceCategory(" ".join(this_window), p, tokenizer, loaded_model))
            tmp_result = np.argmax(tmp_result_list)
            if tmp_result == 1 and not re.search(u'[bBcCdDfFgGjJlLqQsSvVxXyYzZ]', " ".join(this_window)) and not re.search(u'[āēīōūĀĒĪŌŪ]', " ".join(this_window)):
                result.extend([tmp_result_list for _ in range(w)])
            elif tmp_result == 2 and not re.search(u'[āēīōūĀĒĪŌŪ]', " ".join(this_window)):
                result += [tmp_result_list for _ in range(w)]
            else:
                if w >= 4 and w % 2 == 0:
                    result += detectCodeSwitchingPointDynamicWindowVersion(" ".join(this_window), w-2, tokenizer, loaded_model)
                elif w > 1:
                    result += detectCodeSwitchingPointDynamicWindowVersion(" ".join(this_window), w-1, tokenizer, loaded_model)
                else:
                    result += detectCodeSwitchingPointDynamicWindowVersion(" ".join(this_window), w, tokenizer, loaded_model)
            ptr += w
        return result
## End of BiLSTM model ##

## MBERT model ##
@torch.no_grad()
def sentenceCategoryMbertVersion(text: str, model):
    tokenized_text = mbert_tokenizer(text, padding="longest", truncation=True, return_tensors='pt')
    prediction = model(input_ids=tokenized_text["input_ids"], attention_mask=tokenized_text["attention_mask"], token_type_ids=tokenized_text["token_type_ids"])
    return list(torch.nn.functional.softmax(prediction.logits, dim=-1).detach().cpu().numpy()[0])

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
            return [[0, 1, 0]]
        elif re.search(u'[bBcCdDfFgGjJlLqQsSvVxXyYzZ]', x):
            return [[0, 0, 1]]
        else:
            return [sentenceCategoryMbertVersion(x, model)]

    elif end == 2:
        tmp_result_list = sentenceCategoryMbertVersion(x, model)
        tmp_result = np.argmax(tmp_result_list)
        if tmp_result == 1 and not re.search(u'[āēīōūĀĒĪŌŪ]', x) and not re.search(u'[bBcCdDfFgGjJlLqQsSvVxXyYzZ]', x):
            return [tmp_result_list, tmp_result_list]
        else:
            return detectCodeSwitchingPointMbertVersion(words_list[0], 1, model) + detectCodeSwitchingPointMbertVersion(words_list[1], 1, model)
    
    else:
        result = []
        ptr = 0
        while ptr < end:
            this_window = words_list[ptr:ptr+w]
            if ptr + w > end:
                w = end - ptr
            else:
                pass
            tmp_result_list = list(sentenceCategoryMbertVersion(" ".join(this_window), model))
            tmp_result = np.argmax(tmp_result_list)
            if tmp_result == 1 and not re.search(u'[bBcCdDfFgGjJlLqQsSvVxXyYzZ]', " ".join(this_window)) and not re.search(u'[āēīōūĀĒĪŌŪ]', " ".join(this_window)):
                result.extend([tmp_result_list for _ in range(w)])
            elif tmp_result == 2 and not re.search(u'[āēīōūĀĒĪŌŪ]', " ".join(this_window)):
                result += [tmp_result_list for _ in range(w)]
            else:
                if w >= 4 and w % 2 == 0:
                    result += detectCodeSwitchingPointMbertVersion(" ".join(this_window), w-2, model)
                elif w > 1:
                    result += detectCodeSwitchingPointMbertVersion(" ".join(this_window), w-1, model)
                else:
                    result += detectCodeSwitchingPointMbertVersion(" ".join(this_window), w, model)
            ptr += w
        return result
## End of Mbert model ##
# End of functions #

output_format = '''{<br/>"input": "Maori English Maori",<br/>"cleaned": "Maori English Maori",<br/>"labels": ["M", "E", "M"],<br/>"probability": [[1,0], [0,1], [1,0]]<br/>}'''

avaliable_models = ['size_2_bilstm', 'size_2_bilstm_lower', 'size_3_bilstm', 'size_3_bilstm_lower', 'full_size_bilstm', 'full_size_bilstm_lower', 'size_2_mbert', 'size_2_mbert_lower', 'size_3_mbert', 'size_3_mbert_lower', 'full_size_mbert', 'full_size_mbert_lower']

@app.route('/favicon.ico', methods=['GET'])
def favicon():
    return redirect("https://aotearoavoices.nz/favicon.ico")

@app.route('/getModel', methods=['GET'])
def getModel():
        # return '<title>M/E CW Detection API</title>Avaliable Models are: {}.<br/><br/>For more information, please visit <a href="./getInfo">getInfo</a>.<div style="position:fixed;bottom:0;background-color:white;width:100%"><div style="text-align:center"><label style="padding-right:4px;">Copyright © 2022</label><a href="https://speechresearch.auckland.ac.nz/">Speech Research Group @ UoA</a><label>. All rights reserved.</label></div></div>'.format(", ".join(avaliable_models))
        return jsonify({"models": avaliable_models, "usage": "At /getInfo page", "more information": "https://openreview.net/forum?id=rAxl_GibSWq"})

@app.route('/getInfo', methods=['GET'])
def getInfo():
    return '<title>M/E CW Detection API</title>This API is used to detect the code switching point between English and Māori.<br/><br/>Query format: /?model=&ltmodel_name&gt&text=&lttext&gt<br/><br/>Response JSON format:<br/><br/>{}<br/><br/>where E is English, M is Māori<br/><br/>To get the list of models, visit <a href="./getModel">getModel</a>.<br/><br/>To get the usage and the purpose of the API, visit <a href="./getInfo">getInfo</a>.<r/><br/><br/>For more information of the definition of code switch detection and the models used, please visit: <a href="https://openreview.net/forum?id=rAxl_GibSWq">https://openreview.net/forum?id=rAxl_GibSWq</a><div style="position:fixed;bottom:0;background-color:white;width:100%"><div style="text-align:center"><label style="padding-right:4px;">Copyright © 2022</label><a href="https://speechresearch.auckland.ac.nz/">Speech Research Group @ UoA</a><label>. All rights reserved.</label></div></div>'.format(output_format)

@app.errorhandler(404)
def not_found(error):
    return redirect('/getInfo') # Modify to other route

@app.route('/', methods=['GET'])
def detect():
    args = request.args
    if 'model' not in args or 'text' not in args:
        return redirect('/getInfo') # Modify to other route
    else:
        model_name = args['model']
        input_text = args['text']
        cleaned_text = cleanText(input_text).replace('-', ' ')
        if model_name == "size_2_bilstm":
            prediction_result = detectCodeSwitchingPointDynamicWindowVersion(cleaned_text, 2, size_2_bilstm_tokenizer, size_2_bilstm_model)
        elif model_name == "size_2_bilstm_lower":
            prediction_result = detectCodeSwitchingPointDynamicWindowVersion(cleaned_text.lower(), 2, size_2_bilstm_lower_tokenizer, size_2_bilstm_lower_model)

        elif model_name == "size_3_bilstm":
            prediction_result = detectCodeSwitchingPointDynamicWindowVersion(cleaned_text, 3, size_3_bilstm_tokenizer, size_3_bilstm_model)
        elif model_name == "size_3_bilstm_lower":
            prediction_result = detectCodeSwitchingPointDynamicWindowVersion(cleaned_text.lower(), 3, size_3_bilstm_lower_tokenizer, size_3_bilstm_lower_model)
        
        elif model_name == "full_size_bilstm":
            prediction_result = detectCodeSwitchingPointDynamicWindowVersion(cleaned_text, 250, full_size_bilstm_tokenizer, full_size_bilstm_model)
        elif model_name == "full_size_bilstm_lower":
            prediction_result = detectCodeSwitchingPointDynamicWindowVersion(cleaned_text.lower(), 250, full_size_bilstm_lower_tokenizer, full_size_bilstm_lower_model)
        
        elif model_name == "size_2_mbert":
            prediction_result = detectCodeSwitchingPointMbertVersion(cleaned_text, 2, size_2_mbert_model)
        elif model_name == "size_2_mbert_lower":
            prediction_result = detectCodeSwitchingPointMbertVersion(cleaned_text.lower(), 2, size_2_mbert_lower_model)
        
        elif model_name == "size_3_mbert":
            prediction_result = detectCodeSwitchingPointMbertVersion(cleaned_text, 3, size_3_mbert_model)
        elif model_name == "size_3_mbert_lower":
            prediction_result = detectCodeSwitchingPointMbertVersion(cleaned_text.lower(), 3, size_3_mbert_lower_model)
        
        elif model_name == "full_size_mbert":
            prediction_result = detectCodeSwitchingPointMbertVersion(cleaned_text, 4, full_size_mbert_model)
        elif model_name == "full_size_mbert_lower":
            prediction_result = detectCodeSwitchingPointMbertVersion(cleaned_text.lower(), 4, full_size_mbert_lower_model)
        
        else:
            return redirect('/getInfo') # Modify to other route

        ### Output ###
        print(prediction_result)
        outdict = {}
        outdict['input'] = input_text
        outdict['cleaned'] = cleaned_text
        outdict['labels'] = [ 'M' if np.argmax(item[1:], axis=0) == 0 else 'E' for item in prediction_result ]
        outdict['probability'] = [ [f'{item[1]:.2f}', f'{item[2]:.2f}'] for item in prediction_result ]
        return jsonify(outdict)

if __name__ == '__main__':
    # Bind 127.0.0.1:8500, only accept connections from localhost
    # app.run(host='127.0.0.1', port=8500, debug=True)
    app.run(host='0.0.0.0', port=8500, debug=True)
    