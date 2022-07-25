import pandas as pd

WINDOW_SIZE = 3

FILE_NAME = "20220321_Hansard_DB_MP_only.csv"

df = pd.read_csv(FILE_NAME)

df.drop(columns=['id', 'number', 'Labels_Final'], inplace=True)

newDf = pd.DataFrame(columns=['text', 'label'])
tmpDf = pd.DataFrame(columns=['text', 'label'])

length = len(df)

for row in df.itertuples():
    text = row.text.split(' ')
    label = row.label.split(',')
    if len(text) > len(label):
        text = text[:len(label)]
    elif len(text) < len(label):
        label = label[:len(text)]
    else:
        pass
    
    print(f'Progress: {row.Index}/{length}', end='\t\t\t\t\t\t\t\t\r')

    for i in range(0, len(text), WINDOW_SIZE):
        # newDf = pd.concat([newDf, pd.DataFrame({'text': ' '.join(text[i:i+WINDOW_SIZE]), 'label': ','.join(label[i:i+WINDOW_SIZE])}, index=[0])], ignore_index=True)
        tmpDfLen = len(tmpDf)
        if tmpDfLen < 5001:
            tmpDf.loc[tmpDfLen] = [' '.join(text[i:i+WINDOW_SIZE]), ','.join(label[i:i+WINDOW_SIZE])]
        else:
            newDf = pd.concat([newDf, tmpDf], ignore_index=True)
            tmpDf = pd.DataFrame(columns=['text', 'label'])


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

newDf['Labels_Final'] = newDf['label'].apply(labelGen)

newDf.to_csv(FILE_NAME.split('.')[0]+'_window_'+str(WINDOW_SIZE)+'.csv', index=False)
