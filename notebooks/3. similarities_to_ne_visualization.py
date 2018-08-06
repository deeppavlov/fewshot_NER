
# coding: utf-8

# In[290]:


import copy
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_hub as hub
import deeppavlov
from deeppavlov.dataset_readers.ontonotes_reader import OntonotesReader
from deeppavlov.models.preprocessors.capitalization import CapitalizationPreprocessor
from deeppavlov.models.embedders.glove_embedder import GloVeEmbedder


# ### Read data

# In[291]:


reader = OntonotesReader()
dataset = reader.read(data_path='data/')
print(dataset.keys())
print('Num of train sentences: {}'.format(len(dataset['train'])))
print('Num of test sentences: {}'.format(len(dataset['test'])))


# ### Drop train sentences with no named entities

# In[292]:


dataset_sanitized = []
for example in dataset['train']:
    tags = example[1]
    if any(map(lambda t: t != 'O', tags)):
        dataset_sanitized.append(example)
dataset['train'] = dataset_sanitized
print('Num of train sentences with at least one NE: {}'.format(len(dataset['train'])))


# ### Select few examples

# In[293]:


n_examples = 5
# np.random.seed(12)
indices = np.random.choice(len(dataset['train']), size=n_examples)
examples = [dataset['train'][i] for i in indices]
print(examples)


# ### Split tokens and tags

# In[294]:


def split_tokens_tags(dataset: list):
    tokens = []
    tags = []
    for sample in dataset:
        tokens.append(sample[0])
        tags.append(sample[1])
    return tokens, tags


# In[295]:


tokens_train,tags_train = split_tokens_tags(examples)
# print(tags_train)


# ### Elmo wrapper class

# In[296]:


class ElmoEmbedder():
    def __init__(self):
        self.elmo = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        self.sess = sess

    def get_tokens_embeddings(self, tokens_input: list, tokens_length:list=None):
        if not tokens_length:
            if isinstance(tokens_input[0], list):
                tokens_length = [len(seq) for seq in tokens_input]
            else:
                tokens_length = len(tokens_input)
        embeddings = self.elmo(
                        inputs={
                            "tokens": tokens_input,
                            "sequence_len": tokens_length
                        },
                        signature="tokens",
                        as_dict=True)["elmo"]
        embeddings = self.sess.run([embeddings])
        return embeddings[0]


# ### Utility functions

# In[297]:


def tags2binary(tags, symb=True):
    tags = copy.deepcopy(tags)
    for seq in tags:
        for i in range(len(seq)):
            if symb:
                if seq[i] != 'O':
                    seq[i] = 'T'
            else:
                seq[i] = 1 if seq[i] != 'O' else 0
    return tags


# In[298]:


def to_lower_case(tokens:list):
    tokens_lower = []
    for seq in tokens:
        tokens_lower.append([])
        for token in seq:
            tokens_lower[-1].append(token.lower())
    return tokens_lower


# In[299]:


def get_tokens_len(tokens):
    if isinstance(tokens[0], str):
        tokens = [tokens]
    return [len(seq) for seq in tokens]


# In[300]:


def add_padding(tokens:list):
    if isinstance(tokens[0], str):
        return tokens, len(tokens)
    elif isinstance(tokens[0], list):
        tokens = copy.deepcopy(tokens)
        max_len = 0
        for seq in tokens:
            if len(seq) > max_len:
                max_len = len(seq)
        for seq in tokens:
            i = len(seq)
            while i < max_len:
                seq.append('')
                i += 1
        return tokens
    else:
        raise Exception('tokens should be either list of strings or list of lists of strings')


# In[301]:


def flatten_list(ar:list):
    flat = []
    for sublist in ar:
        flat += sublist
    return flat


# In[302]:


def select_list_elements(ar:list, indices:list):
    return [ar[i] for i in indices]


# ### Transform NER to binary classification

# In[303]:


tags_train = tags2binary(tags_train)


# ### Count how many named entities in support set

# In[304]:


tokens_train_flat = flatten_list(tokens_train)
tags_train_flat = flatten_list(tags_train)
ne_train_count = 0
for tag in tags_train_flat:
    if tag == 'T':
        ne_train_count += 1
print('Count of tokens in support set {}'.format(len(tokens_train_flat)))
print('Count of named entities in support set {}'.format(ne_train_count))
print('ratio #NE/#tokens = {:.4f}'.format(ne_train_count/len(tokens_train_flat)))


# ### Main embedder

# In[305]:


class CompositeEmbedder():
    def __init__(self, use_elmo=True, elmo_scale=1., use_cap_feat=False, use_glove=False):
        self.use_elmo = use_elmo
        self.elmo_scale = elmo_scale
        self.use_cap_feat = use_cap_feat
        self.use_glove = use_glove
        if self.use_elmo:
            self.elmo = ElmoEmbedder()
        if self.use_cap_feat:
            self.cap_prep = CapitalizationPreprocessor()
        if self.use_glove:
            self.glove = GloVeEmbedder('embeddings/glove.6B/glove.6B.100d.txt', pad_zero=True)
        
    def embed(self, tokens: list):
        if isinstance(tokens[0], str):
            tokens = [tokens]
        # Get ELMo embeddings
        if self.use_elmo:
            tokens_input = add_padding(tokens)
            tokens_length = get_tokens_len(tokens)
            embeddings = self.elmo.get_tokens_embeddings(tokens_input, tokens_length)
            embeddings *= self.elmo_scale
            embed_size = embeddings.shape[-1]
#             print(embeddings.shape)
#             print(embed_size)

        # Use capitalization features
        if self.use_cap_feat:
#             print('Use capitalization features')
            cap_features = self.cap_prep(tokens)
    #         print(cap_features)
#             print(cap_features.shape)
            embeddings = np.concatenate((embeddings, cap_features), axis=2)
            embed_size = embeddings.shape[-1]
#             print(embeddings.shape)

        # Use GloVe embeddings
        if self.use_glove:
#             print('Use GloVe')
            
            glove_embed = self.glove(to_lower_case(tokens))
            glove_embed = np.array(glove_embed)
            if not self.use_elmo:
                embeddings = glove_embed
            else: 
                embeddings = np.concatenate((embeddings, glove_embed), axis=2)
            embed_size = embeddings.shape[-1]
#             print(embeddings.shape)
        
        return embeddings


# ### Calculate similarity of token embedding vector to some prototype (centroid) or just support vector

# In[306]:


def calc_sim(token_vec, support_vec)->dict:
    sim = {}
    sim['euc_dist'] = np.linalg.norm(token_vec - support_vec)
    sim['dot_prod'] = np.dot(token_vec, support_vec)
    sim['cosine'] = np.dot(token_vec, support_vec)/(np.linalg.norm(token_vec)*np.linalg.norm(support_vec)) if np.linalg.norm(support_vec) != 0 else 0
    return sim


# In[307]:


def calc_sim_batch(tokens: list, embeddings: np.ndarray, support_vec: np.ndarray)->list:
    sim_list = []
    tokens_length = get_tokens_len(tokens)
    for i in range(len(tokens_length)):
        sim_list.append([])
        for j in range(tokens_length[i]):
            token_vec = embeddings[i,j,:]
            sim_list[i].append(calc_sim(token_vec, support_vec))
    return sim_list


# In[308]:


def flatten_sim(sim_list):
    sims_flat = {'euc_dist': [], 'dot_prod': [], 'cosine': []}
    for i in range(len(sim_list)):
        for j in range(len(sim_list[i])):
            for sim_type in ['euc_dist', 'dot_prod', 'cosine']:
                sims_flat[sim_type].append(sim_list[i][j][sim_type])
    for sim_type in ['euc_dist', 'dot_prod', 'cosine']:
        sims_flat[sim_type] = np.array(sims_flat[sim_type])
    return sims_flat


# ### Calculate centroid for named entities embedding vectors

# In[309]:


def calc_ne_centroid_vec(tokens: list, tags: list, embeddings: np.ndarray=None, embedder: CompositeEmbedder=None):

    # Calculate embeddings
    if embedder != None:
        embeddings = embedder.embed(tokens)

    # Calculate average vector for ne-tags
    embed_size = embeddings.shape[-1]
    ne_prototype = np.zeros((embed_size,))
    tokens_length = get_tokens_len(tokens)
    tags_bin = np.array(flatten_list(tags2binary(tags, symb=False)))
#     print(tags_bin)
    n_ne_tags = np.sum(tags_bin == 1)
    embeddings_ne_flat = np.zeros((n_ne_tags, embed_size))
#     print(n_ne_tags)
#     n_ne_tags = 0
    k = 0
    for i in range(len(tokens_length)):
        for j in range(tokens_length[i]):
            if tags[i][j] == 'T':
                ne_prototype += embeddings[i,j,:].reshape((embed_size,))
                embeddings_ne_flat[k,:] = embeddings[i,j,:]
                k += 1   # TODO: Change this to some better approach
#                 n_ne_tags += 1
    if n_ne_tags != 0:
        ne_prototype /= n_ne_tags
#     print('ne mean vector: {}'.format(ne_prototype))
    
    # Calculate similarities
    sim_list = calc_sim_batch(tokens, embeddings, ne_prototype)

    return ne_prototype, sim_list, embeddings, embeddings_ne_flat


# ### Calculate similarities of some test tokens to NE prototype

# In[310]:


def calc_sim_to_ne_prototype(tokens: list, ne_prototype: np.ndarray, embeddings: np.ndarray=None, embedder: CompositeEmbedder=None):
    if isinstance(tokens[0], str):
        tokens = [tokens]
    
    tokens_length = get_tokens_len(tokens)
    
    # Calculate embeddings
    if embedder != None:
        embeddings = embedder.embed(tokens)
    
    # Calculate similarities
    sim_list = calc_sim_batch(tokens, embeddings, ne_prototype)
    
    return sim_list, embeddings


# In[311]:


def calc_sim_to_ne_nearest(tokens: list, ne_support_embeddings: np.ndarray, embeddings: np.ndarray=None, embedder: CompositeEmbedder=None):
    if isinstance(tokens[0], str):
        tokens = [tokens]
    
    tokens_length = get_tokens_len(tokens)
    
    # Calculate embeddings
    if embedder != None:
        embeddings = embedder.embed(tokens)
    
    # Calculate similarities
    n_supports = ne_support_embeddings.shape[0]
    sim_list = []
    tokens_length = get_tokens_len(tokens)
    for i in range(len(tokens_length)):
        sim_list.append([])
        for j in range(tokens_length[i]):
            token_vec = embeddings[i,j,:]
            sim_token_list = {'euc_dist': [], 'dot_prod': [], 'cosine': []}
            for k in range(n_supports):
                sim = calc_sim(token_vec, ne_support_embeddings[k, :])
                for sim_type in ['euc_dist', 'dot_prod', 'cosine']:
                    sim_token_list[sim_type].append(sim[sim_type])
            sim_list[i].append({'euc_dist': np.min(np.array(sim_token_list['euc_dist'])), 
                                'dot_prod': np.max(np.array(sim_token_list['dot_prod'])), 
                                'cosine': np.max(np.array(sim_token_list['cosine']))})
            
    return sim_list, embeddings
    


# ### Embedder initialisation

# In[312]:


embedder = CompositeEmbedder(use_elmo=True, elmo_scale=1, use_cap_feat=True, use_glove=True)


# ### Calculate NE centroid for support examples

# In[313]:


ne_prototype, _, _, ne_support_embeddings = calc_ne_centroid_vec(tokens_train, tags_train, embedder=embedder)
print(ne_support_embeddings.shape)


# ### Select some unlabeled examples from the test set and estimate similarity to a named entity for each token

# In[314]:


n_test_sentences = 100
# np.random.seed(44)
indices_test = np.random.choice(len(dataset['test']), size=n_test_sentences)
test_sentences = select_list_elements(dataset['test'], indices_test)
tokens_test,tags_test = split_tokens_tags(test_sentences)
# print(tokens_test)
# print(tags_test)


# In[315]:


tags_test_bin_flat = np.array(flatten_list(tags2binary(tags_test, symb=False)))


# In[316]:


METHOD = 'NEAREST' #'PROTOTYPE'
if METHOD == 'PROTOTYPE':
    sim_list_test, _ = calc_sim_to_ne_prototype(tokens_test, ne_prototype, embedder=embedder)
elif METHOD == 'NEAREST':
    sim_list_test, _ = calc_sim_to_ne_nearest(tokens_test, ne_support_embeddings, embedder=embedder)


# ### Group similarities with tokens

# In[317]:


def zip_tokens_sim(tokens: list, sim_list: list, sim_type='cosine'):
    tokens_sim = []
    for i in range(len(tokens)):
        tokens_sim.append([])
        for j in range(len(tokens[i])):
            tokens_sim[-1].append((tokens[i][j], sim_list[i][j][sim_type]))
    return tokens_sim


# In[318]:


tokens_test_sim = zip_tokens_sim(tokens_test, sim_list_test)


# ### Print test sentences with NE similarities estimations

# In[319]:


def decorate_ne_token(token, tag):
    if tag == 'T':
        token = '[[' + token + ']]'
    return token
def format_labeled_examples(tokens_input: list, tags_input: list):
    s = '+++++++++ Input examples +++++++++\n\n'
    for i in range(len(tokens_input)):
        for j in range(len(tokens_input[i])):
            s += decorate_ne_token(tokens_input[i][j], tags_input[i][j]) + ' '
        s += '\n\n'
    return s


# In[320]:


def format_inference_results(tokens_sim: list):
    s = '+++++++++ Tests +++++++++\n\n'
    for seq in tokens_sim:
        for token, sim in seq:
            s += '{}[{:.3f}]'.format(token, sim)
            s += ' '
        s += '\n\n'
    return s


# In[321]:


text = ''
text += format_labeled_examples(tokens_train, tags_train)
text += format_inference_results(tokens_test_sim)
print(text)


# ### Visualize similarities of tokens to NE

# In[322]:


from IPython.core.display import display, HTML


# In[323]:


def get_color(red=0, green=255, blue=0):
    return {'r': red, 'g': green, 'b': blue}

def get_rgba_str(color, alpha=1):
    return 'rgba({},{},{},{})'.format(color['r'], color['g'], color['b'], alpha)

def get_token_span_str(token, color, cf=1):
    return '<span style="padding: 0.15em; margin-right: 4px; border-radius: 0.25em; background: {};">{}</span>'.format(get_rgba_str(color, alpha=cf), token)

def wrap_with_style(html):
    return '<div style="line-height: 1.5em;">{:s}</div>'.format(html)


# In[324]:


bg_color = get_color(red=0, green=255, blue=0)
display(HTML(get_token_span_str('token', bg_color, cf=0.7)))


# In[325]:


sim_flat = flatten_sim(sim_list_test)
sim_min = np.min(sim_flat['cosine'])
sim_max = np.max(sim_flat['cosine'])


# In[326]:


def sim_transform_lin(sim):
    # similarity transformation for better visualization
    return (sim - sim_min)/(sim_max - sim_min)
def sim_transform(sim, T=0.5):
    # similarity transformation with temperature for better visualization
    return (np.exp(sim/T) - np.exp(sim_min/T))/(np.exp(sim_max/T) - np.exp(sim_min/T))


# In[398]:


def get_colored_results_html(tokens_sim: list, color, T=0.5):
    s = '<h3 style="margin-bottom:0.3em;">Visualization of tokens to NE similarities on test set:</h3>'
    for seq in tokens_sim:
        for token, sim in seq:
            s += get_token_span_str(token, color, cf=sim_transform(sim, T))
#             s += ' '
        s += '<br/><br/>'
    return wrap_with_style(s)


# In[328]:


display(HTML(get_colored_results_html(tokens_test_sim, bg_color)))


# ### Plot histograms of test similarities

# In[329]:


sim_flat = np.array(flatten_sim(sim_list_test)['cosine'])
sim_test_words = sim_flat[tags_test_bin_flat == 0]
sim_test_ne = sim_flat[tags_test_bin_flat == 1]
print(sim_test_words.shape)
print(sim_test_ne.shape)


# In[330]:


plt.hist(sim_test_words, color = 'green', edgecolor = 'black',
         bins = 30, label='words')
plt.hist(sim_test_ne, color = 'red', edgecolor = 'black',
         bins = 30, label='named entities')
plt.legend()
plt.title('Histograms')


# ### Plot probability densities

# In[331]:


# https://stackoverflow.com/questions/15415455/plotting-probability-density-function-by-sample-with-matplotlib
from scipy.stats.kde import gaussian_kde
from numpy import linspace
kde_words = gaussian_kde( sim_test_words )
dist_space_words = linspace( min(sim_test_words), max(sim_test_words), 100 )
kde_ne = gaussian_kde( sim_test_ne )
dist_space_ne = linspace( min(sim_test_ne), max(sim_test_ne), 100 )
plt.plot( dist_space_words, kde_words(dist_space_words), color='green', label='words' )
plt.plot( dist_space_ne, kde_ne(dist_space_ne), color='red',  label='named entities' )
plt.legend()
plt.grid()
plt.title('Probability densities')


# ### Interactive demo

# In[402]:


print('>> Input examples placing NE inside square brackets. Input "exit" at the end')
inp = ''
inputs = []
while inp != 'q':
    inp = input('-> ')
    if inp != 'q':
        inputs.append(inp)
print(inputs)


# In[403]:


import re
def findNE(sentences:list):
    ne_list = []
    sentences_sanitized = []
    pattern = re.compile(r'\[([a-zA-Z]+)\]')
    for sent in sentences:
        ne_list.append([])
        for ne in pattern.findall(sent):
            ne_list[-1].append(ne)
        sent = sent.replace('[', '')
        sent = sent.replace(']', '')
        sentences_sanitized.append(sent)
    return ne_list, sentences_sanitized
ne_list, inputs_clean = findNE(inputs)
print(inputs_clean)
print(ne_list)


# In[404]:


from deeppavlov.models.tokenizers.spacy_tokenizer import StreamSpacyTokenizer
from deeppavlov.models.tokenizers.nltk_tokenizer import NLTKTokenizer
tokenizer = NLTKTokenizer()
tokens = tokenizer(inputs_clean)
tags = []
for i, sent in enumerate(tokens):
    tags.append([])
    for token in sent:
        tag = 'T' if token in ne_list[i] else 'O'
        tags[-1].append(tag)
print(tokens)
print(tags)


# In[405]:


ne_prototype, _, _, ne_support_embeddings = calc_ne_centroid_vec(tokens, tags, embedder=embedder)
print(ne_support_embeddings.shape)


# In[406]:


print('>> Input test sentences. Input "exit" at the end')
inp_test = ''
inputs_test = []
while inp_test != 'q':
    inp_test = input('-> ')
    if inp_test != 'q':
        inputs_test.append(inp_test)
print(inputs_test)


# In[407]:


tokens_test = tokenizer(inputs_test)
print(tokens_test)


# In[411]:


METHOD = 'PROTOTYPE' #'PROTOTYPE'
if METHOD == 'PROTOTYPE':
    sim_list_test, _ = calc_sim_to_ne_prototype(tokens_test, ne_prototype, embedder=embedder)
elif METHOD == 'NEAREST':
    sim_list_test, _ = calc_sim_to_ne_nearest(tokens_test, ne_support_embeddings, embedder=embedder)


# In[412]:


tokens_test_sim = zip_tokens_sim(tokens_test, sim_list_test)
display(HTML(get_colored_results_html(tokens_test_sim, bg_color, T=1)))


# In[420]:


display(HTML('<form><input type="checkbox"><button type="submit">Submit</button></form>'))

