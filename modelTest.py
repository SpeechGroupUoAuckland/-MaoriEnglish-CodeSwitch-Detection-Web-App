import torch
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
import json
import pandas as pd
import numpy as np
import re
import pickle
import tensorflow as tf
from keras.utils.data_utils import pad_sequences
import numpy as np

# Read test set
test = pd.read_csv('20220321_Hansard_DB_test_MP_only.csv')
# only take the rows with Processed = 'Y'
test = test[test['Processed'] == 'Y']

base_folder = './web/models/'
# Parameters #
full_size_bilstm_model_path = base_folder+'bilstm.h5'
full_size_bilstm_tokenizer_path = base_folder+'tokenizerBilstm.pickle'
full_size_bilstm_lower_model_path = base_folder+'bilstmLower.h5'
full_size_bilstm_lower_tokenizer_path = base_folder+'tokenizerBilstmLower.pickle'

size_2_bilstm_model_path = base_folder+'bilstmSize2.h5'
size_2_bilstm_tokenizer_path = base_folder+'tokenizerBilstmSize2.pickle'
size_2_bilstm_lower_model_path = base_folder+'bilstmSize2Lower.h5'
size_2_bilstm_lower_tokenizer_path = base_folder+'tokenizerBilstmSize2Lower.pickle'

size_3_bilstm_model_path = base_folder+'bilstmSize3.h5'
size_3_bilstm_tokenizer_path = base_folder+'tokenizerBilstmSize3.pickle'
size_3_bilstm_lower_model_path = base_folder+'bilstmSize3Lower.h5'
size_3_bilstm_lower_tokenizer_path = base_folder+'tokenizerBilstmSize3Lower.pickle'

full_size_mbert_model_path = base_folder+'mbert'
full_size_mbert_lower_model_path = base_folder+'mbertLower'

mbert_tokenizer_path = base_folder+'tokenizerMbert'
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
    predict = loaded_model.predict(padded, verbose = 0) 
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
                elif w > 1:
                    result += detectCodeSwitchingPointDynamicWindowVersion(" ".join(this_window), w-1, tokenizer, loaded_model)
                else:
                    result += detectCodeSwitchingPointDynamicWindowVersion(" ".join(this_window), w, tokenizer, loaded_model)
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
                elif w > 1:
                    result += detectCodeSwitchingPointMbertVersion(" ".join(this_window), w-1, model)
                else:
                    result += detectCodeSwitchingPointMbertVersion(" ".join(this_window), w, model)
            ptr += w
        return result
## End of Mbert model ##

def transfrom(a: list) -> list:
    for index, item in enumerate(a):
        if item == 1:
            a[index] = 'M'
        elif item == 2:
            a[index] = 'P'
        else:
            a[index] = 'U'
    return a
# End of functions #

# Test
def test_model(model_name, window_size):
    model = globals()[f'{model_name}_model']
    tokenizer = globals()[f'{model_name}_tokenizer']

    lower_flag = True if 'lower' in model_name else False

    word_count = 0
    wrong_word_count = 0
    wrong_word_dict = {} # {word1: count1, word2: count2, ...}

    sentence_count = 0
    wrong_sentence_count = 0
    wrong_sentence_list = [] # [row.id1, row.id2, ...]

    for row in test.itertuples():
        text = row.text.lower() if lower_flag else row.text
        predict = transfrom(detectCodeSwitchingPointDynamicWindowVersion(text, window_size, tokenizer, model))
        real = list(row.label)
        if len(predict) == len(real):
            wrong_sentence = False
            word_count += len(real)
            for index, item in enumerate(predict):
                if item != real[index]:
                    wrong_sentence = True
                    wrong_word_count += 1
                    current_word = row.text.split()[index].lower()
                    if current_word not in wrong_word_dict:
                        wrong_word_dict[current_word] = 1
                    else:
                        wrong_word_dict[current_word] += 1
                else:
                    pass
            if wrong_sentence:
                wrong_sentence_count += 1
                wrong_sentence_list.append(row.id)
            else:
                pass
            sentence_count += 1
        else:
            # print("Error: length of predict and real is not equal,", row.id)
            pass

    # Save the wrong words and wrong sentences
    with open(f"evaluation/{model_name}_error_dict.json", "w") as f:
        f.write('{\n'+f'"{wrong_word_count}/{word_count}":')
        json.dump(wrong_word_dict, f)
        f.write(f',\n"{wrong_sentence_count}/{sentence_count}":')
        json.dump(wrong_sentence_list, f)
        f.write('\n}')


test_model('full_size_bilstm', 250)
test_model('full_size_bilstm_lower', 250)

test_model('size_2_bilstm', 2)
test_model('size_2_bilstm_lower', 2)

test_model('size_3_bilstmd', 3)
test_model('size_3_bilstm_lower', 3)
