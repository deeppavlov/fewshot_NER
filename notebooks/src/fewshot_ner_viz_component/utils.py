import numpy as np
import copy
from sklearn.metrics import f1_score
from collections import OrderedDict
import math
import matplotlib.pyplot as plt
from scipy.stats.kde import gaussian_kde
from numpy import linspace
import re

# Utility functions
def get_tokens_len(tokens):
    if isinstance(tokens[0], str):
        tokens = [tokens]
    return [len(seq) for seq in tokens]

def to_lower_case(tokens:list):
    tokens_lower = []
    for seq in tokens:
        tokens_lower.append([])
        for token in seq:
            tokens_lower[-1].append(token.lower())
    return tokens_lower

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

def calc_sim(token_vec, support_vec)->dict:
    sim = {}
    sim['euc_dist'] = np.exp(-np.linalg.norm(token_vec - support_vec, axis=-1))
    sim['dot_prod'] = np.dot(token_vec, support_vec)
    sim['cosine'] = np.dot(token_vec, support_vec)/(np.linalg.norm(token_vec)*np.linalg.norm(support_vec)) if np.linalg.norm(support_vec) != 0 else 0
    return sim

def calc_euc_dist(x1: np.ndarray, x2: np.ndarray):
    return np.linalg.norm(x1 - x2, axis=-1)

def calc_mahalanobis_dist(v: np.ndarray, X: np.ndarray):
    X_cov = np.cov(X, rowvar=False)
    X_mean = np.mean(X, axis=0)
    # d = np.zeros(v.shape[0])
    # for i in range(v.shape[0]):
    #     v_e = v[i, :]
    #     d[i] = np.sqrt(np.dot(np.dot((v_e - X_mean), np.linalg.pinv(X_cov)), (v_e - X_mean).T))
    D = v - X_mean
    X_cov_inv = np.linalg.pinv(X_cov)
    # return np.sqrt(np.sum(np.dot(D, X_cov_inv)*D, axis=1))
    return np.sum(np.dot(D, X_cov_inv)*D, axis=1)

def normalize(x: np.ndarray):
    e = 1e-10
    return x/(np.tile(np.expand_dims(np.linalg.norm(x, axis=-1), axis=-1), x.shape[-1]) + e)

def calc_sim_by_type(x1: np.ndarray, x2: np.ndarray, sim_type='cosine'):
    if sim_type == 'euc_dist':
        return np.exp(-np.linalg.norm(x1 - x2, axis=-1))
    elif sim_type == 'dot_prod':
        return np.dot(x1, x2.T)
    elif sim_type == 'cosine':
        return np.dot(normalize(x1), normalize(x2).T)
    elif sim_type == 'mahalanobis':
        if len(x2.shape) < 2 or x2.shape[1] < 2:
            raise Exception('x2 have to be a matrix')
        return np.exp(-calc_mahalanobis_dist(x1, x2))

def calc_sim_batch(tokens: list, embeddings: np.ndarray, support_vec: np.ndarray)->list:
    sim_list = []
    tokens_length = get_tokens_len(tokens)
    for i in range(len(tokens_length)):
        sim_list.append([])
        for j in range(tokens_length[i]):
            token_vec = embeddings[i,j,:]
            sim_list[i].append(calc_sim(token_vec, support_vec))
    return sim_list

def calc_sim_ne_centroid(X_support, y_support, X_query, sim_type='cosine'):
    X_sup_ne = X_support[y_support == 1, :]
    X_sup_words = X_support[y_support == 0, :]
    ne_sup_centroid = np.mean(X_sup_ne, axis=0)
    sim_q_list = calc_sim_by_type(X_query, ne_sup_centroid, sim_type)
    return sim_q_list

def plotPDE(sims, y, info=''):
    sims_words = sims[y == 0]
    sims_ne = sims[y == 1]
    plt.figure(figsize=(7,5))
    kde_words = gaussian_kde(sims_words)
    dist_space_words = linspace(min(sims_words), max(sims_words), 100)
    kde_ne = gaussian_kde( sims_ne )
    dist_space_ne = linspace(min(sims_ne), max(sims_ne), 100)
    plt.plot( dist_space_words, kde_words(dist_space_words), color='green', label='words' )
    plt.plot( dist_space_ne, kde_ne(dist_space_ne), color='red',  label='named entities' )
    plt.legend(loc='upper right')
    plt.title(info)
    plt.grid()

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
def plot_tSNE(X, y: np.ndarray, colors=('g', 'r'), labels=('words','entities'), title='', use_pca=False, subplot=None):
    X = deepcopy(X)
    if use_pca:
        X = PCA(n_components=50).fit_transform(X)
    tsne = TSNE(n_components=2, method='exact', init='pca')
    X_2d = tsne.fit_transform(X)
    if not subplot:
        plt.figure()
    else:
        plt.subplot(subplot['nrows'], subplot['ncols'], subplot['index'], title=title)
    for i in range(2):
        X_sel = X_2d[y == i, :]
        plt.scatter(X_sel[:, 0], X_sel[:, 1], c=colors[i], alpha= 0.5, label=labels[i])
    plt.legend()
    if not subplot:
        plt.title(title)

def get_tokens_count(tokens:list):
    return len([t for seq in tokens for t in seq])
def embeddings2feat_mat(embeddings:np.ndarray, tokens_length):
    n_tokens = sum(tokens_length)
    n_features = embeddings.shape[-1]
    feat_mat = np.zeros((n_tokens, n_features))
#     print(feat_mat.shape)
    k = 0
    for i in range(len(tokens_length)):
        for j in range(tokens_length[i]):
            feat_mat[k, :] = embeddings[i, j, :]
            k += 1
    return feat_mat

def flatten_list(ar:list):
    flat = []
    for sublist in ar:
        flat += sublist
    return flat

def getNeTagMainPart(tag:str):
    return tag[2:] if tag != 'O' else tag

def tags2binaryFlat(tags):
    return np.array([1 if t == 'T' or (len(t) > 2 and t[2:] == 'T') else 0 for seq in tags for t in seq])

def tags2binaryPadded(tags:list):
    if isinstance(tags[0], str):
        tags = [tags]
    n_sentences = len(tags)
    tokens_length = get_tokens_len(tags)
    max_len = np.max(tokens_length)
    tokens_length = np.tile(np.expand_dims(tokens_length, -1), (1,max_len))
    y = np.zeros((n_sentences, max_len))
    range_ar = np.tile(np.arange(1, max_len+1, 1), (n_sentences, 1))
    for i, sen in enumerate(tags):
        for j, tag in enumerate(sen):
            if tags[i][j] != 'O':
                y[i][j] = 1
#     y[range_ar > tokens_length] = -1
    return y

def get_matrices(tokens, tags, embedder):
    return (embeddings2feat_mat(embedder.embed(tokens), get_tokens_len(tokens)),
           tags2binaryFlat(tags))

def removeBIOFromTags(tags):
    tags_res = []
    for sen in tags:
        tags_res.append([])
        for tag in sen:
            tag_norm = tag
            if len(tag) > 2:
                tag_norm = tag[2:]
            tags_res[-1].append(tag_norm)
    return tags_res

def predToTags(pred, accountBIO=False):
    pred = copy.deepcopy(pred)
    if isinstance(pred, list):
        pred = np.array(pred)
    pred_tags = ['O']*pred.size
    for i in range(pred.size):
        if pred[i] == 1:
            pred_tags[i] = 'T'
            if accountBIO:
                if i > 0 and pred[i-1] == 1:
                    pred_tags[i] = 'I-T'
                else:
                    pred_tags[i] = 'B-T'
    return pred_tags

def flatten_sim(sim_list):
    sims_flat = {'euc_dist': [], 'dot_prod': [], 'cosine': []}
    for i in range(len(sim_list)):
        for j in range(len(sim_list[i])):
            for sim_type in ['euc_dist', 'dot_prod', 'cosine']:
                sim = sim_list[i][j].get(sim_type)
                if sim != None:
                    sims_flat[sim_type].append(sim)
    for sim_type in ['euc_dist', 'dot_prod', 'cosine']:
        sims_flat[sim_type] = np.array(sims_flat[sim_type])
    return sims_flat


# ### Group similarities with tokens
def zip_tokens_sim(tokens: list, sim_list: list, sim_type='cosine'):
    tokens_sim = []
    for i in range(len(tokens)):
        tokens_sim.append([])
        for j in range(len(tokens[i])):
            tokens_sim[-1].append((tokens[i][j], sim_list[i][j][sim_type]))
    return tokens_sim

def zip_tokens_sim_list(tokens, sim_list):
    tokens_sim = []
    k = 0
    # print(len(sim_list.shape))
    for seq in tokens:
        tokens_sim.append([])
        for t in seq:
            tokens_sim[-1].append((t, sim_list[k]))
            k += 1
    return tokens_sim

def flat_sim_one_type(sim_list: list, sim_type: str):
    sims_flat = []
    for i in range(len(sim_list)):
        for j, sim_group in enumerate(sim_list[i]):
            sim = None
            if isinstance(sim_group, dict) and sim_group.get(sim_type) != None:
                sim = sim_group[sim_type]
            elif isinstance(sim_group, float) or isinstance(sim_group, int):
                sim = sim_group
            sims_flat.append(sim)
    return sims_flat

def calc_sim_min_max(sim_list, single_metric=False):
    if single_metric:
        sim_flat = flatten_list(sim_list)
    else:
        sim_flat = flatten_sim(sim_list)['cosine']
    sim_min = np.min(sim_flat)
    sim_max = np.max(sim_flat)
    return (sim_min, sim_max)

def sim_transform(sim, sim_min, sim_max, T=0.5):
    # similarity transformation with temperature for better visualization
    return (np.exp(sim/T) - np.exp(sim_min/T))/(np.exp(sim_max/T) - np.exp(sim_min/T))

def infer_tags(sim_list, sim_type, T=0.5, threshold=0.5):
    sim_min, sim_max = calc_sim_min_max(sim_list)
    tokens_length = get_tokens_len(sim_list)
    tags = [['T' if sim_transform(sim_list[i][j][sim_type], sim_min, sim_max, T)  > threshold else 'O' for j in range(tokens_length[i])] for i in range(len(tokens_length))]
    return tags

def split_tokens_tags(dataset: list):
    tokens = []
    tags = []
    for sample in dataset:
        tokens.append(sample[0])
        tags.append(sample[1])
    return tokens, tags

def calc_data_props(tokens:list, tags:list):
    props = {}
    props['ne_types'] = {}
    tokens_flat = flatten_list(tokens)
    tags_flat = flatten_list(tags)
    ne_count = 0
    for tag in tags_flat:
        if tag != 'O':
            ne_count += 1
            tag_main = tag[2:]
            if props['ne_types'].get(tag_main) != None:
                props['ne_types'][tag_main] += 1
            else:
                props['ne_types'][tag_main] = 1
    props['sent_count'] = len(tokens)
    props['tokens_count'] = len(tokens_flat)
    props['ne_count'] = ne_count
    props['ne_ratio'] = props['ne_count']/props['tokens_count']
    for k in props['ne_types'].keys():
        props['ne_types'][k] /= ne_count

    return props

def print_data_props(props:dict):
    s = ''
    s += '#sentences = {}, '.format(props['sent_count'])
    s += '#tokens = {}, '.format(props['tokens_count'])
    s += '#ne = {}, '.format(props['ne_count'])
    s += '#ne / #tokens = {:.3f}, '.format(props['ne_ratio'])
    print(s)

def softmax(ar, scale=True):
    ar = ar[:]
    eps = 1e-10
    if scale:
        ar_min = np.min(ar)
        ar_max = np.max(ar)
        if abs(ar_max - ar_min) > eps:
            ar = (ar - ar_min)/(ar_max - ar_min)
    return np.exp(ar)/(np.sum(np.exp(ar)))

def pred_class_labels_bin(scores: np.ndarray, threshold: float):
    scores = copy.deepcopy(scores)
    if isinstance(scores, list):
        scores = np.array(scores)
    pred = np.zeros(scores.size, dtype=int)
    pred[scores >= threshold] = 1
    return pred

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

def chunk_finder(current_token, previous_token, tag):
    current_tag = current_token.split('-', 1)[-1]
    previous_tag = previous_token.split('-', 1)[-1]
    if previous_tag != tag:
        previous_tag = 'O'
    if current_tag != tag:
        current_tag = 'O'
    if (previous_tag == 'O' and current_token == 'B-' + tag) or \
            (previous_token == 'I-' + tag and current_token == 'B-' + tag) or \
            (previous_token == 'B-' + tag and current_token == 'B-' + tag) or \
            (previous_tag == 'O' and current_token == 'I-' + tag):
        create_chunk = True
    else:
        create_chunk = False

    if (previous_token == 'I-' + tag and current_token == 'B-' + tag) or \
            (previous_token == 'B-' + tag and current_token == 'B-' + tag) or \
            (current_tag == 'O' and previous_token == 'I-' + tag) or \
            (current_tag == 'O' and previous_token == 'B-' + tag):
        pop_out = True
    else:
        pop_out = False
    return create_chunk, pop_out

def precision_recall_f1_chunks(y_true, y_pred, print_results=True, short_report=False, entity_of_interest=None):
    # Find all tags
    tags = set()
    for tag in y_true + y_pred:
        if tag != 'O':
            current_tag = tag[2:]
            tags.add(current_tag)
    tags = sorted(list(tags))

    results = OrderedDict()
    for tag in tags:
        results[tag] = OrderedDict()
    results['__total__'] = OrderedDict()
    n_tokens = len(y_true)
    total_correct = 0
    # Firstly we find all chunks in the ground truth and prediction
    # For each chunk we write starting and ending indices

    for tag in tags:
        count = 0
        true_chunk = list()
        pred_chunk = list()
        y_true = [str(y) for y in y_true]
        y_pred = [str(y) for y in y_pred]
        prev_tag_true = 'O'
        prev_tag_pred = 'O'
        while count < n_tokens:
            yt = y_true[count]
            yp = y_pred[count]

            create_chunk_true, pop_out_true = chunk_finder(yt, prev_tag_true, tag)
            if pop_out_true:
                true_chunk[-1].append(count - 1)
            if create_chunk_true:
                true_chunk.append([count])

            create_chunk_pred, pop_out_pred = chunk_finder(yp, prev_tag_pred, tag)
            if pop_out_pred:
                pred_chunk[-1].append(count - 1)
            if create_chunk_pred:
                pred_chunk.append([count])
            prev_tag_true = yt
            prev_tag_pred = yp
            count += 1

        if len(true_chunk) > 0 and len(true_chunk[-1]) == 1:
            true_chunk[-1].append(count - 1)
        if len(pred_chunk) > 0 and len(pred_chunk[-1]) == 1:
            pred_chunk[-1].append(count - 1)

        # Then we find all correctly classified intervals
        # True positive results
        tp = 0
        for start, stop in true_chunk:
            for start_p, stop_p in pred_chunk:
                if start == start_p and stop == stop_p:
                    tp += 1
                if start_p > stop:
                    break
        total_correct += tp
        # And then just calculate errors of the first and second kind
        # False negative
        fn = len(true_chunk) - tp
        # False positive
        fp = len(pred_chunk) - tp
        if tp + fp > 0:
            precision = tp / (tp + fp) * 100
        else:
            precision = 0
        if tp + fn > 0:
            recall = tp / (tp + fn) * 100
        else:
            recall = 0
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0
        results[tag]['precision'] = precision
        results[tag]['recall'] = recall
        results[tag]['f1'] = f1
        results[tag]['n_predicted_entities'] = len(pred_chunk)
        results[tag]['n_true_entities'] = len(true_chunk)
    total_true_entities = 0
    total_predicted_entities = 0
    total_precision = 0
    total_recall = 0
    total_f1 = 0
    for tag in results:
        if tag == '__total__':
            continue
        n_pred = results[tag]['n_predicted_entities']
        n_true = results[tag]['n_true_entities']
        total_true_entities += n_true
        total_predicted_entities += n_pred
        total_precision += results[tag]['precision'] * n_pred
        total_recall += results[tag]['recall'] * n_true
        total_f1 += results[tag]['f1'] * n_true
    if total_true_entities > 0:
        accuracy = total_correct / total_true_entities * 100
        total_recall = total_recall / total_true_entities
    else:
        accuracy = 0
        total_recall = 0
    if total_predicted_entities > 0:
        total_precision = total_precision / total_predicted_entities
    else:
        total_precision = 0

    if total_precision + total_recall > 0:
        total_f1 = 2 * total_precision * total_recall / (total_precision + total_recall)
    else:
        total_f1 = 0

    results['__total__']['n_predicted_entities'] = total_predicted_entities
    results['__total__']['n_true_entities'] = total_true_entities
    results['__total__']['precision'] = total_precision
    results['__total__']['recall'] = total_recall
    results['__total__']['f1'] = total_f1

    if print_results:
        s = 'processed {len} tokens ' \
            'with {tot_true} phrases; ' \
            'found: {tot_pred} phrases;' \
            ' correct: {tot_cor}.\n\n'.format(len=n_tokens,
                                              tot_true=total_true_entities,
                                              tot_pred=total_predicted_entities,
                                              tot_cor=total_correct)

        s += 'precision:  {tot_prec:.2f}%; ' \
             'recall:  {tot_recall:.2f}%; ' \
             'FB1:  {tot_f1:.2f}\n\n'.format(acc=accuracy,
                                             tot_prec=total_precision,
                                             tot_recall=total_recall,
                                             tot_f1=total_f1)

        if not short_report:
            for tag in tags:
                if entity_of_interest is not None:
                    if entity_of_interest in tag:
                        s += '\t' + tag + ': precision:  {tot_prec:.2f}%; ' \
                                          'recall:  {tot_recall:.2f}%; ' \
                                          'F1:  {tot_f1:.2f} ' \
                                          '{tot_predicted}\n\n'.format(tot_prec=results[tag]['precision'],
                                                                       tot_recall=results[tag]['recall'],
                                                                       tot_f1=results[tag]['f1'],
                                                                       tot_predicted=results[tag]['n_predicted_entities'])
                elif tag != '__total__':
                    s += '\t' + tag + ': precision:  {tot_prec:.2f}%; ' \
                                      'recall:  {tot_recall:.2f}%; ' \
                                      'F1:  {tot_f1:.2f} ' \
                                      '{tot_predicted}\n\n'.format(tot_prec=results[tag]['precision'],
                                                                   tot_recall=results[tag]['recall'],
                                                                   tot_f1=results[tag]['f1'],
                                                                   tot_predicted=results[tag]['n_predicted_entities'])
        elif entity_of_interest is not None:
            s += '\t' + entity_of_interest + ': precision:  {tot_prec:.2f}%; ' \
                              'recall:  {tot_recall:.2f}%; ' \
                              'F1:  {tot_f1:.2f} ' \
                              '{tot_predicted}\n\n'.format(tot_prec=results[entity_of_interest]['precision'],
                                                           tot_recall=results[entity_of_interest]['recall'],
                                                           tot_f1=results[entity_of_interest]['f1'],
                                                           tot_predicted=results[entity_of_interest]['n_predicted_entities'])
        print(s)
    return results
