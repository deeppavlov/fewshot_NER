import numpy as np
import sys
import copy
from math import ceil, floor
from sklearn.svm import SVC
import tensorflow_hub as hub
import tensorflow as tf
from deeppavlov.dataset_readers.ontonotes_reader import OntonotesReader
from deeppavlov.models.preprocessors.capitalization import CapitalizationPreprocessor
from deeppavlov.models.embedders.glove_embedder import GloVeEmbedder
from src.fewshot_ner_viz_component.utils import *
# from utils import *

class FewshotNerBinaryClassifier():
    def __init__(self, embedder):
        self.embedder = embedder
        self.X_train = None
        self.y_train = None
        self.n_ne_tags = 0
        self.ne_prototype = None
        self.embeddings_ne_flat = None
        self.svm_clf = SVC(probability=True, kernel='sigmoid')

    def train_on_batch(self, tokens: list, tags: list):
        print('Train')
        # Calculate embeddings
        embeddings = self.embedder.embed(tokens)

        # Calculate average vector for ne-tags
        embed_size = embeddings.shape[-1]
        if self.ne_prototype != None:
            ne_prototype = self.ne_prototype*self.n_ne_tags
        else:
            ne_prototype = np.zeros((embed_size,))

        tokens_length = get_tokens_len(tokens)
        tags_bin = tags2binaryFlat(tags)
    #     print(tags_bin)
        n_ne_tags = np.sum(tags_bin == 1) + self.n_ne_tags
        embeddings_ne_flat = np.zeros((n_ne_tags, embed_size))
    #     print(n_ne_tags)
    #     n_ne_tags = 0
        k = 0
        for i in range(len(tokens_length)):
            for j in range(tokens_length[i]):
                if tags[i][j] == 'T' or tags[i][j] == 1:
                    ne_prototype += embeddings[i,j,:].reshape((embed_size,))
                    embeddings_ne_flat[k,:] = embeddings[i,j,:]
                    k += 1
    #                 n_ne_tags += 1
        if n_ne_tags != 0:
            ne_prototype /= n_ne_tags
        # print('ne mean vector: {}'.format(ne_prototype))

        # Calculate similarities
        # sim_list = calc_sim_batch(tokens, embeddings, ne_prototype)

        X_train = embeddings2feat_mat(embeddings, get_tokens_len(tokens))
        y_train = tags_bin

        self.n_ne_tags = n_ne_tags
        self.ne_prototype = ne_prototype
        if self.embeddings_ne_flat != None:
            self.embeddings_ne_flat = np.vstack([self.embeddings_ne_flat, embeddings_ne_flat])
        else:
            self.embeddings_ne_flat = embeddings_ne_flat

        if self.X_train !=None:
            self.X_train = np.vstack([self.X_train, X_train])
            self.y_train = np.vstack([self.y_train, y_train])
        else:
            self.X_train = X_train
            self.y_train = y_train

        # SVM train
        n_ne = sum(self.y_train)
        n_words = self.y_train.size - sum(self.y_train)
        n_tokens = n_ne + n_words
        weights = [n_tokens/(2*n_ne) if label == 1 else n_tokens/(2*n_words) for label in self.y_train]
        self.svm_clf.fit(self.X_train, self.y_train, weights)

    def predict(self, tokens, methods, params):
        if isinstance(methods,str):
            methods = [methods]
        embeddings = self.embedder.embed(tokens)
        X_test = embeddings2feat_mat(embeddings, get_tokens_len(tokens))
        results = {}
        if 'ne_centroid' in methods:
            results.update({
                'ne_centroid': self._predict_with_ne_centroid(tokens, embeddings, **params['ne_centroid'])})
        if 'ne_nearest' in methods:
            results.update({
                'ne_nearest': self._predict_with_ne_nearest(tokens, embeddings, **params['ne_nearest'])})
        if 'svm' in methods:
            results.update({
                'svm': self._predict_with_SVM(X_test)})
        if 'weighted_kNN' in methods:
            results.update({
                'weighted_kNN': self._predict_with_weighted_kNN(X_test, **params['weighted_kNN'])})
        if 'centroid_kNN' in methods:
            results.update({
                'centroid_kNN': self._predict_with_centroid_kNN(X_test, **params['centroid_kNN'])})

        return results

    def _predict_with_ne_centroid(self, tokens: list, embeddings: np.ndarray=None, sim_type='cosine'):
        print('NE centroid similarity model')
        if isinstance(tokens[0], str):
            tokens = [tokens]

        tokens_length = get_tokens_len(tokens)
        if not isinstance(embeddings, np.ndarray) and embeddings == None:
            embeddings = self.embedder.embed(tokens)

        # Calculate similarities
        sim_list = calc_sim_batch(tokens, embeddings, self.ne_prototype)

        # Predict classes
        sim_min, sim_max = calc_sim_min_max(sim_list)
        probas = np.array([sim_transform(sim_list[i][j][sim_type], sim_min, sim_max, T=0.5) for i in range(len(tokens_length)) for j in range(tokens_length[i])])
        pred = tags2binaryFlat(infer_tags(sim_list, sim_type, T=1, threshold=0.5))

        return {'sim': sim_list, 'pred': pred, 'probas': probas}

    def _predict_with_ne_nearest(self, tokens: list, embeddings: np.ndarray = None, sim_type='cosine'):
        print('NE nearest similarity model')
        if isinstance(tokens[0], str):
            tokens = [tokens]

        tokens_length = get_tokens_len(tokens)

        if not isinstance(embeddings, np.ndarray) and embeddings == None:
            embeddings = self.embedder.embed(tokens)

        # Calculate similarities
        ne_support_embeddings = self.embeddings_ne_flat
        n_supports = ne_support_embeddings.shape[0]
        sim_list = []
        tokens_length = get_tokens_len(tokens)
        for i in range(len(tokens_length)):
            sim_list.append([])
            for j in range(tokens_length[i]):
                token_vec = embeddings[i,j,:]
                sim_token_list = []
                for k in range(n_supports):
                    sim = calc_sim(token_vec, ne_support_embeddings[k, :])[sim_type]
                    sim_token_list.append(sim)
                sim_list[i].append({sim_type: np.max(np.array(sim_token_list))})

        # Predict classes
        sim_min, sim_max = calc_sim_min_max(sim_list)
        probas = np.array([sim_transform(sim_list[i][j][sim_type], sim_min, sim_max, T=0.5) for i in range(len(tokens_length)) for j in range(tokens_length[i])])
        pred = tags2binaryFlat(infer_tags(sim_list, sim_type, T=1, threshold=0.5))

        return {'sim': sim_list, 'pred': pred, 'probas': probas}

    def _predict_with_SVM(self, X_test: np.ndarray):
        print('SVM classifier model')
        pred = self.svm_clf.predict(X_test)
        probas = self.svm_clf.predict_proba(X_test)[:,1]

        return {'pred': pred, 'probas': probas}

    def _predict_with_weighted_kNN(self, X_test, k=3, metric='dot_prod', use_class_weights=False, use_sim_weights=True):
        print('Weighted kNN model')
        print('k = {}, metric: {}'. format(k, metric))
        X_train = self.X_train
        y_train = self.y_train
        # Weights for classes
        n_classes = np.unique(y_train).size
        n_train_samples = y_train.size
        class_weights = np.array([1,1])
        if use_class_weights:
            class_weights = n_train_samples/(n_classes*np.bincount(y_train))
    #     print(np.bincount(y_train))
    #     print(n_classes)
    #     print(weights)
        n_test_samples = X_test.shape[0]
        probas = np.zeros((n_test_samples))
        pred = np.zeros((n_test_samples), dtype=np.int64)
        # Find k nearest neighbours for each test sample
        for idx_test in range(n_test_samples):
            x = X_test[idx_test,:]
            top_k_sim = np.zeros((k))
            top_k_sim.fill(np.NINF)
            top_k_labels = np.zeros((k), dtype=np.int64)
            for idx_train in range(n_train_samples):
                sim = calc_sim(x, X_train[idx_train, :])[metric]
                for i, sim_from_top in enumerate(top_k_sim):
                    if sim > sim_from_top:
                        top_k_sim[i] = sim
                        top_k_labels[i] = y_train[idx_train]
                        break
    #         print(top_k_sim)
    #         print(top_k_labels)
            calc_prob_dist = lambda ar: ar/(np.sum(ar))
            n_labels_c1 = np.sum(top_k_labels.astype(np.int64))
            n_labels_c0 = top_k_labels.size - n_labels_c1
            if use_sim_weights:
                n_labels_c1 = top_k_sim[top_k_labels == 1].dot(np.ones((n_labels_c1)))
                n_labels_c0 = top_k_sim[top_k_labels == 0].dot(np.ones((n_labels_c0)))
            bincount = np.array([n_labels_c0, n_labels_c1])
    #         print(bincount)
            prob_cur = calc_prob_dist(class_weights*bincount)
    #         print(prob_cur)
            probas[idx_test] = prob_cur[1]
            pred[idx_test] = 1 if prob_cur[1] > prob_cur[0] else 0
    #         print(pred[idx_test])
            # https://stackoverflow.com/questions/3002085/python-to-print-out-status-bar-and-percentage
            sys.stdout.write('\r')
            progress = idx_test/X_test.shape[0]
            sys.stdout.write("[%-20s] %d%%" % ('='*int(ceil(progress*20)), ceil(progress*100)))
            sys.stdout.flush()
    #         print(prob_cur)
    #     print(probas)
        print()
        return {'pred': pred, 'probas': probas}

    def _predict_with_centroid_kNN(self, X_test, y_test=None, k=5, metric='cosine', use_class_weights=False):
        print('NE centroid + words kNN similarity model')
        print('k = {}, metric: {}'. format(k, metric))
        X_train = self.X_train
        y_train = self.y_train
        # Weights for classes
        n_classes = np.unique(y_train).size
        n_train_samples = y_train.size
        class_weights = np.array([1,1])
        if use_class_weights:
            class_weights = n_train_samples/(n_classes*np.bincount(y_train))

        # Centroid for class 1 examples
        centroid_c1 = np.mean(X_train[y_train == 1, :], axis=0)
    #     print(centroid_c1.shape)

        n_test_samples = X_test.shape[0]
        probas = np.zeros((n_test_samples))
        pred = np.zeros((n_test_samples), dtype=np.int64)
        # Find k nearest neighbours of class 0 for each test sample
        for idx_test in range(n_test_samples):
            x = X_test[idx_test,:]
            sim_c1 = calc_sim(x, centroid_c1)[metric]
            top_k_sim = np.zeros((k))
            top_k_sim.fill(np.NINF)
            for _, example_c0 in enumerate(X_train[y_train == 0, :]):
                sim = calc_sim(x, example_c0)[metric]
                for i, sim_from_top in enumerate(top_k_sim):
                    if sim > sim_from_top:
                        top_k_sim[i] = sim
                        break
    #         print(top_k_sim)
    #         print(top_k_labels)
            sim_c0 = np.mean(top_k_sim)
    #         print(sim_c0)
    #         print(sim_c1)
            prob_cur = softmax(class_weights*np.array([sim_c0, sim_c1]))
    #         print(prob_cur)
            probas[idx_test] = prob_cur[1]
            pred[idx_test] = 1 if prob_cur[1] > prob_cur[0] else 0
    #         print(pred[idx_test])
            # https://stackoverflow.com/questions/3002085/python-to-print-out-status-bar-and-percentage
            sys.stdout.write('\r')
            progress = idx_test/X_test.shape[0]
            sys.stdout.write("[%-20s] %d%%" % ('='*int(ceil(progress*20)), ceil(progress*100)))
            sys.stdout.flush()
    #         print(prob_cur)
    #     print(probas)
        print()
        return {'pred': pred, 'probas': probas}

class ElmoEmbedder():
    def __init__(self, custom_weights=False, weights: list = []):
        self.custom_weights = custom_weights
        self.weights = weights
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
        elmo_res = self.elmo(
                        inputs={
                            "tokens": tokens_input,
                            "sequence_len": tokens_length
                        },
                        signature="tokens",
                        as_dict=True)
        embeddings_sum = elmo_res["elmo"]
        word_emb = elmo_res["word_emb"]
        embeddings_layer1 = elmo_res["lstm_outputs1"]
        embeddings_layer2 = elmo_res["lstm_outputs2"]
        embeddings_sum, word_emb, embeddings_layer1, embeddings_layer2 = self.sess.run([embeddings_sum, word_emb, embeddings_layer1, embeddings_layer2])
        word_emb = np.concatenate((word_emb, word_emb), axis=-1)
        # print(embeddings_sum.shape)
        # print(embeddings_layer1.shape)
        # print(embeddings_layer2.shape)
        embeddings = embeddings_sum
        if self.custom_weights:
            if len(self.weights) == 2:
                embeddings = embeddings_layer1*self.weights[0] + embeddings_layer2*self.weights[1]
            elif len(self.weights) == 3:
                embeddings = word_emb*self.weights[0] + embeddings_layer1*self.weights[1] + embeddings_layer2*self.weights[2]
        return embeddings

class CompositeEmbedder():
    def __init__(self, use_elmo=True, elmo_scale=1., cap_scale=1., use_cap_feat=False, use_glove=False, elmo_params={}):
        self.use_elmo = use_elmo
        self.elmo_scale = elmo_scale
        self.cap_scale = cap_scale
        self.use_cap_feat = use_cap_feat
        self.use_glove = use_glove
        if self.use_elmo:
            self.elmo = ElmoEmbedder(**elmo_params)
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
            cap_features = self.cap_prep(tokens)*self.cap_scale
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
