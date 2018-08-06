import copy
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_hub as hub
import deeppavlov
from deeppavlov.models.preprocessors.capitalization import CapitalizationPreprocessor
from deeppavlov.models.embedders.glove_embedder import GloVeEmbedder

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

def to_lower_case(tokens:list):
    tokens_lower = []
    for seq in tokens:
        tokens_lower.append([])
        for token in seq:
            tokens_lower[-1].append(token.lower())
    return tokens_lower

def get_tokens_len(tokens):
    if isinstance(tokens[0], str):
        tokens = [tokens]
    return [len(seq) for seq in tokens]

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

def flatten_list(ar:list):
    flat = []
    for sublist in ar:
        flat += sublist
    return flat

def select_list_elements(ar:list, indices:list):
    return [ar[i] for i in indices]

def calc_sim(token_vec, support_vec)->dict:
    sim = {}
    sim['euc_dist'] = np.linalg.norm(token_vec - support_vec)
    sim['dot_prod'] = np.dot(token_vec, support_vec)
    sim['cosine'] = np.dot(token_vec, support_vec)/(np.linalg.norm(token_vec)*np.linalg.norm(support_vec)) if np.linalg.norm(support_vec) != 0 else 0
    return sim

def calc_sim_batch(tokens: list, embeddings: np.ndarray, support_vec: np.ndarray)->list:
    sim_list = []
    tokens_length = get_tokens_len(tokens)
    for i in range(len(tokens_length)):
        sim_list.append([])
        for j in range(tokens_length[i]):
            token_vec = embeddings[i,j,:]
            sim_list[i].append(calc_sim(token_vec, support_vec))
    return sim_list

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

# ### Group similarities with tokens
def zip_tokens_sim(tokens: list, sim_list: list, sim_type='cosine'):
    tokens_sim = []
    for i in range(len(tokens)):
        tokens_sim.append([])
        for j in range(len(tokens[i])):
            tokens_sim[-1].append((tokens[i][j], sim_list[i][j][sim_type]))
    return tokens_sim

# ### Print test sentences with NE similarities estimations
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

def format_inference_results(tokens_sim: list):
    s = '+++++++++ Tests +++++++++\n\n'
    for seq in tokens_sim:
        for token, sim in seq:
            s += '{}[{:.3f}]'.format(token, sim)
            s += ' '
        s += '\n\n'
    return s
