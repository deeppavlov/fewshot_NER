import copy
from deeppavlov.dataset_readers.ontonotes_reader import OntonotesReader
from src.fewshot_ner_viz_component.utils import *
# from utils import *

def read_data():
    reader = OntonotesReader()
    dataset = reader.read(data_path='data/')
    # print(dataset.keys())
    print('Num of train sentences: {}'.format(len(dataset['train'])))
    print('Num of valid sentences: {}'.format(len(dataset['valid'])))
    print('Num of test sentences: {}'.format(len(dataset['test'])))
    print(dataset['train'][50:60])
    return dataset

def filter_data_by_ne_type(data:list, ne_types:list, tags2binary=False, preserveBIO=False, keepIfAny=True):
    if ne_types == None or len(ne_types) == 0:
        return data
    data_filtered = []
    for tokens,tags in data:
        contains_all = True
        contains_any = False
        tags_norm = [getNeTagMainPart(t) for t in tags]
        for ne_type in ne_types:
            if not ne_type in tags_norm:
                contains_all = False
            if ne_type in tags_norm:
                contains_any = True
        if contains_all or (keepIfAny and contains_any):
            if tags2binary:
                if preserveBIO:
                    tags = [tags[i][:2]+'T' if t in ne_types else 'O' for i,t in enumerate(tags_norm)]
                else:
                    tags = ['T' if t in ne_types else 'O' for t in tags_norm]
            data_filtered.append((tokens,tags))
    return data_filtered

def filter_dataset_by_ne_types(dataset: list, ne_types, preserveBIO=False, keepIfAny=True):
    dataset = copy.deepcopy(dataset)
    if not isinstance(ne_types, list):
        ne_types = [ne_types]
    for dataset_type in ['train', 'valid', 'test']:
        dataset[dataset_type] = filter_data_by_ne_type(dataset[dataset_type], ne_types, preserveBIO=preserveBIO, tags2binary=True)
        print('Num of {} sentences: {}'.format(dataset_type, len(dataset[dataset_type])))
    return dataset

def get_data_sample(data, n_samples: int):
    indices = np.random.choice(len(data), size=n_samples, replace=False)
    return split_tokens_tags([data[i] for i in indices])
