import pandas as pd

RAW_TEST = '20220321_Hansard_DB_test_only.csv'

df = pd.read_csv(RAW_TEST)

df['text'] = df['text'].astype(str)
df['label'] = df['label'].astype(str)
df['Labels_Final'] = df['Labels_Final'].astype(str)

# only process bilinguals
df = df[df['Labels_Final'] == 'B']

# remove the rows whose label is not A or M or P only
df = df[df['label'].map(lambda x: all([char in ['A', 'M', 'P', ','] for char in list(x)]))]

# remove the rows whose label has more than one A
df = df[df['label'].map(lambda x: x.count('A') <= 1)]

def covertRule(label: str) -> str:
    label_list = label.split(',')
    
    if 'P' in label_list and 'M' in label_list:
        max_idx = len(label_list) - 1
        for index, item in enumerate(label_list):
            if item == 'A':
                # Edges
                if index == 0:
                    if label_list[1] == 'P':
                        return ''.join([char if char != 'A' else 'P' for char in label_list])
                    elif label_list[1] == 'M':
                        return ''.join([char if char != 'A' else 'M' for char in label_list])
                    else:
                        return ''.join(label_list) # ERROR case, should not happen
                elif index == max_idx:
                    if label_list[-2] == 'P':
                        return ''.join([char if char != 'A' else 'P' for char in label_list])
                    elif label_list[-2] == 'M':
                        return ''.join([char if char != 'A' else 'M' for char in label_list])
                    else:
                        return ''.join(label_list) # ERROR case, should not happen
                
                # Middle
                if label_list[index - 1] == 'P' and label_list[index + 1] == 'P':
                    return ''.join([char if char != 'A' else 'P' for char in label_list])
                elif label_list[index - 1] == 'M' and label_list[index + 1] == 'M':
                    return ''.join([char if char != 'A' else 'M' for char in label_list])
                else:
                    return ''.join(label_list) # leave for manual process

            else:
                pass

    # NEVER HAPPEN 
    elif 'P' in label_list:
        return ''.join([char if char != 'A' else 'P' for char in label_list])

    elif 'M' in label_list:
        return ''.join([char if char != 'A' else 'M' for char in label_list])
    
    else:
        return ''.join(label_list) # ERROR case, should not happen

df['label'] = df['label'].apply(covertRule)

df['Processed'] = df['label'].apply(lambda x: 'N' if 'A' in x else 'Y')

df.to_csv('20220321_Hansard_DB_test_MP_only.csv', index=False)
