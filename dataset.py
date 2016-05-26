import os
import pandas as pd

def create_dataset(folder, save_name):
    subfolders = [('neg', 0), ('pos', 1)]
    index = []
    dataset = []
    for subfolder, label in subfolders:
        path = os.path.join(folder, subfolder)
        filenames = sorted(os.listdir(path))
        for filename in filenames:
            with open(os.path.join(path, filename)) as f:
                name = '{}'.format(filename[:filename.find('.txt')])
                index.append(name)
                dataset.append({'sentiment': label, 'review': f.read()})
    df = pd.DataFrame(dataset, index=index)
    df.to_csv(save_name, index_label='id')

create_dataset('data/aclImdb/train/', 'data/imdb/train_data.csv')
create_dataset('data/aclImdb/test/', 'data/imdb/test_data.csv')
