import re
import pandas as pd


def labelGen(text: str) -> str:
    """Read labels and generate the final label"""
    if "P" in text and "M" in text:
        return "B"  # Bilingual
    elif "P" in text:
        return "P"  # English or English + numbers
    elif "M" in text:
        return "M"  # Māori or Māori + numbers
    elif "N" in text:
        return "N" # Pure numbers
    else:
        return "U"  # Unknown


MAORI_SPECIAL_CHARS = ["ā", "ē", "ī", "ō", "ū", "Ā", "Ē", "Ī", "Ō", "Ū"]


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


ALL_FILE_NAME = "20220321_Hansard_DB_all.tsv"
MP_FILE_NAME = "20220321_Hansard_DB_MP_only.tsv"

df = pd.read_csv(ALL_FILE_NAME, sep="\t", header=None, names=[
    "id", "number", "text", "label", "Labels_Final"])
df["text"] = df["text"].astype(str).apply(cleanText)
df['label'] = df['label'].astype(str)
df['Labels_Final'] = df['label'].apply(labelGen)

mp_df = pd.read_csv(MP_FILE_NAME, sep="\t", header=None, names=[
    "id", "number", "text", "label", "Labels_Final"])
mp_df["text"] = mp_df["text"].astype(str).apply(cleanText)
mp_df['label'] = mp_df['label'].astype(str)
mp_df['Labels_Final'] = mp_df['label'].apply(labelGen)

# Save the datasets
df.to_csv(ALL_FILE_NAME.split('.')[0]+'.csv', index=False)
mp_df.to_csv(MP_FILE_NAME.split('.')[0]+'.csv', index=False)

# Generate a subset that only contains M and P labels
# If Labels_final is not B, P, M then remove the row
df_mp = df
df_mp = df_mp[df_mp['Labels_Final'] != 'U']
df_mp = df_mp[df_mp['Labels_Final'] != 'N']

df_mp = df_mp[df_mp['label'].map(lambda x: all([char in ['M', 'P', ','] for char in list(x)]))]

df_mp.to_csv(ALL_FILE_NAME.split('_all')[0]+'_gen_MP_only.csv', index=False)

# save the oppsite of the df_mp subset
df_test = df
df_test = df_test[df_test['label'].map(lambda x: any([char not in ['M', 'P', ','] for char in list(x)]))]
df_test.to_csv(ALL_FILE_NAME.split('_all')[0]+'_test_only.csv', index=False)

print('Task completed!\n')