from cleantext import clean
import os
import pandas as pd
import re

def cleanText(text):
    """Clean the text"""
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


data = ''
for dir, _, filenames in os.walk('embeddingTraining'):
    for filename in filenames:
        if filename.endswith('.txt'):
            with open(os.path.join(dir, filename), 'r', encoding='utf-8', errors='ignore') as f:
                data += f.read()

df = pd.read_csv('embeddingTraining/label_per_sentence_Hansard.csv')

data += ' '.join(df['text'])

del df

data = cleanText(data)

data = clean(data,
             fix_unicode=True,
             to_ascii=False,
             lower=True,
             no_line_breaks=True,
             no_urls=True,
             no_emails=True,
             no_phone_numbers=True,
             no_numbers=True,
             no_digits=True,
             no_currency_symbols=True,
             no_punct=True)

# save the cleaned text
with open('embeddingTraining/cleaned_text.txt', 'w') as f:
    f.write(data)



