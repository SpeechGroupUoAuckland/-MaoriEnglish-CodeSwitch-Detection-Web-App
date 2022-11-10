import sys
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

size_2_mbert_model_path = base_folder+'mbertSize2'
size_2_mbert_lower_model_path = base_folder+'mbertSize2Lower'

size_3_mbert_model_path = base_folder+'mbertSize3'
size_3_mbert_lower_model_path = base_folder+'mbertSize3Lower'

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

# Test
def test_model(model_name, window_size):
    model = globals()[f'{model_name}_model']

    if 'mbert' in model_name:
        def detectWrapper(text, wondow_size, model):
            return detectCodeSwitchingPointMbertVersion(text, wondow_size, model)
    else:
        tokenizer = globals()[f'{model_name}_tokenizer']
        def detectWrapper(text, wondow_size, model):
            return detectCodeSwitchingPointDynamicWindowVersion(text, wondow_size, tokenizer, model)

    lower_flag = True if 'lower' in model_name else False

    word_count = 0
    wrong_word_count = 0
    wrong_word_dict = {} # {word1: count1, word2: count2, ...}

    sentence_count = 0
    wrong_sentence_count = 0
    wrong_sentence_list = [] # [row.id1, row.id2, ...]

    maori_mispredicted_as_english = 0
    maori_predicted_as_maori = 0
    english_mispredicted_as_maori = 0
    english_predicted_as_english = 0

    for row in test.itertuples():
        text = row.text.lower() if lower_flag else row.text
        prediction_result = detectWrapper(text, window_size, model)
        predict = [ 'M' if np.argmax(item[1:], axis=0) == 0 else 'P' for item in prediction_result ]
        real = list(row.label)
        if len(predict) == len(real):
            wrong_sentence = False
            word_count += len(real)
            for index, item in enumerate(predict):
                if item != real[index]:
                    if item == 'M':
                        english_mispredicted_as_maori += 1
                    elif item == 'P':
                        maori_mispredicted_as_english += 1
                    else:
                        pass
                    wrong_sentence = True
                    wrong_word_count += 1
                    current_word = row.text.split()[index].lower()
                    if current_word not in wrong_word_dict:
                        wrong_word_dict[current_word] = 1
                    else:
                        wrong_word_dict[current_word] += 1
                else:
                    if item == 'M':
                        maori_predicted_as_maori += 1
                    elif item == 'P':
                        english_predicted_as_english += 1
                    else:
                        pass
            if wrong_sentence:
                wrong_sentence_count += 1
                wrong_sentence_list.append(row.id+f'{row.number}')
            else:
                pass
            sentence_count += 1
        else:
            # print("Error: length of predict and real is not equal,", row.id+row.number)
            pass

    # Save the wrong words and wrong sentences
    with open(f"evaluation/{model_name}_error_dict.json", "w") as f:
        f.write('{\n"confusion_matrix": '+f'[[{maori_predicted_as_maori}, {maori_mispredicted_as_english}], [{english_mispredicted_as_maori}, {english_predicted_as_english}]],\n')
        f.write(f'"{wrong_word_count}/{word_count}":')
        json.dump(wrong_word_dict, f)
        f.write(f',\n"{wrong_sentence_count}/{sentence_count}":')
        json.dump(wrong_sentence_list, f)
        f.write('\n}')

def main():
    args = sys.argv
    if len(args) > 1:
        args = args[1:]
        if args[0] == '-h' or args[0] == '--help':
            print('Usage1: python3 modelTest.py [model_name] [window_size]')
            print('\tExample: python3 modelTest.py mbert 5')
            print('Usage2: python3 -h or python3 --help')
            print('Usage3: python3 -n NUM_CPU')
        elif args[0] == '-n':
            import multiprocessing
            multiprocessing.freeze_support()
            multiprocessing.set_start_method('spawn')
            n_cores = int(args[1])
            pool = multiprocessing.Pool(processes=n_cores)
            pool.starmap(test_model, [
                ('size_2_bilstm', 2),
                ('size_2_bilstm_lower', 2),
                ('size_3_bilstm', 3),
                ('size_3_bilstm_lower', 3),
                ('full_size_bilstm', 250),
                ('full_size_bilstm_lower', 250),
                ('size_2_mbert', 2),
                ('size_2_mbert_lower', 2),
                ('size_3_mbert', 3),
                ('size_3_mbert_lower', 3),
                ('full_size_mbert', 4),
                ('full_size_mbert_lower', 4),
            ])

        else:
            test_model(args[0], int(args[1]))
    else:
        import multiprocessing
        multiprocessing.freeze_support()
        multiprocessing.set_start_method('spawn')
        n_cores = multiprocessing.cpu_count() if multiprocessing.cpu_count() < 12 else 12
        pool = multiprocessing.Pool(processes=n_cores)
        pool.starmap(test_model, [
            ('size_2_bilstm', 2),
            ('size_2_bilstm_lower', 2),
            ('size_3_bilstm', 3),
            ('size_3_bilstm_lower', 3),
            ('full_size_bilstm', 250),
            ('full_size_bilstm_lower', 250),
            ('size_2_mbert', 2),
            ('size_2_mbert_lower', 2),
            ('size_3_mbert', 3),
            ('size_3_mbert_lower', 3),
            ('full_size_mbert', 4),
            ('full_size_mbert_lower', 4),
        ])

if __name__ == '__main__':
    main()
