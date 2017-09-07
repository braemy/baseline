# -*- coding: utf-8 -*-
import json
import os
import pickle
import subprocess

import numpy as np
from tqdm import tqdm
from helper.utils import *
from helper.utils_data import *
from collections import OrderedDict

from helper.FeatureExtractor import FeatureExtractor


class FeatureExtractor_CRF_tf(FeatureExtractor):
    """
    Extracts features from sequence data
    """

    def __init__(self, feature_template, language, embedding_size, pos_tagset):
        FeatureExtractor.__init__(self, feature_template, language, embedding_size)
        # dictionary with all spelling features
        self.spelling_feature_cache = {}
        # feature template used for feature extraction
        self.feature_template = feature_template
        # language of the data set
        self.language = language
        # path to data
        self.data_path = None
        # boolean flag for training
        self.is_training = True
        # dictionary that maps feature string to number
        self._map_feature_str2num = {}
        # dictionary that maps feature number to string
        self._map_feature_num2str = {}
        # dictionary that contains word embeddings (key = word, value = float array)
        self._word_embeddings = None
        self._map_word_index_embedding = {}
        self._embedding_index_counter = 0
        # symbol for unknown word that must be contained in the embeddings and bitstrings dictionaries
        self.unknown_symbol = "<?>"
        self.features_list = []
        self.features_embedding_indexes = []
        self.label_list = []
        self.location_list = []
        self.max_length = -1
        self.pos_tag_list = self.build_pos_mapping(pos_tagset)
        if embedding_size is not None:
            self.embedding_size = int(embedding_size)
            self.embedding_start = np.zeros(self.embedding_size)
            self.embedding_end = np.ones(self.embedding_size) / np.linalg.norm(np.ones(self.embedding_size))

        self.block_size = 50000
        self.quiet = False
        self.embedding_type = None

    def build_pos_mapping(self, postag_list):
        mapping = {p:i for i,p in enumerate(postag_list)}
        mapping["_END_"] = len(mapping)
        mapping["_START_"] = len(mapping)
        return mapping
    def extract_features_crf(self, tokens_list, label_list, pos_list, extract_all, numeric_feature=False):
        """
        Extracts features from the given sequence data.
        @type sequence_data: SequenceData object
        @param sequence_data: contains all word sequences and label sequences
        @type extract_all: bool
        @param extract_all: specifies if features should be extracted for all words or not.  Unless specified
        extract_all=True, it extracts features only from labeled instances
        @type skip_list: list
        @param skip_list: skips extracting features from examples specified by skip_list.
        This is used for active learning. (Pass [] to not skip any example.)
        @return: list of labels, list of features, list of locations (i.e. position in the corpus where each label is
        found)
        """

        self.length = [0] * len(tokens_list)
        # list for labels
        self.label_list = [None] * len(tokens_list)
        # list for features
        self.features_list = [None] * len(tokens_list)
        self.features_embedding_indexes = [None] * len(tokens_list)
        # list for locations
        self.location_list = [None] * len(tokens_list)
        # iterate through all sequences (=sentences) and all words in each sentence
        for sequence_num, (word_sequence, labels_sequence, pos_sequence) in tqdm(
                enumerate(zip(tokens_list, label_list, pos_list)), "Feature extraction", miniters=200, mininterval=2,
                disable=self.quiet):

            sentence_features = [None] * len(labels_sequence)
            sentence_labels = [None] * len(labels_sequence)
            sentence_locations = [None] * len(labels_sequence)
            sentence_embedding_indexes = [None] * len(labels_sequence)

            for position, (label, pos_tags) in enumerate(zip(labels_sequence, pos_sequence)):

                # only use labeled instances unless extract_all=True.
                if (label is not None) or extract_all:
                    # append label id to label list
                    sentence_labels[position] = (self._get_label(label))
                    # append feature id in features list
                    #sentence_features[position] = (
                    #    self._get_features(word_sequence, position, pos_tag=pos_sequence, numeric_feature=numeric_feature))
                    features, embedding_indexes = self._get_features(word_sequence, position, pos_tag=pos_sequence, numeric_feature=numeric_feature)
                    sentence_features[position] = features
                    sentence_embedding_indexes[position] = embedding_indexes

                    # append location in locations list
                    sentence_locations[position] = ((sequence_num, position))

            self.length[sequence_num] = (int(len(word_sequence)))
            self.features_list[sequence_num] = (sentence_features)
            self.features_embedding_indexes[sequence_num] = (sentence_embedding_indexes)
            self.label_list[sequence_num] = (sentence_labels)
            self.location_list[sequence_num] = (sentence_locations)

        return self.features_list, self.features_embedding_indexes, self.label_list, self.length

    def _get_features(self, word_sequence, position, pos_tag=None, numeric_feature=False):
        """
            Finds the integer IDs of the extracted features for a word at a given position in a sequence (=sentence)

            @type word_sequence: list
            @param word_sequence: sequence of words
            @type position: int
            @param position: position in the sequence of words
            @type pos_tag: list
            @param: sequence of pos tag
            @return: a dictionary of numeric features (key = feature id, value = value for the specific feature)
        """
        features, embedding_indexes = self._get_baseline_features(word_sequence, position, pos_tag)
        return features, embedding_indexes



    def _get_baseline_features(self, word_sequence, position, pos_tag_sequence=None):
        """
        Builds the baseline features by using spelling of the word at the position
        and 2 words left and right of the word.

        @type word_sequence: list
        @param word_sequence: sequence of words
        @type position: int
        @param position: position of word in the given sequence
        @type pos_tag_sequence: list
        @param pos_tag_sequence: sequence of pos_tag
        @return: baseline features (dict)
        """
        # get word at given position
        word = get_word(word_sequence, position)

        # extract spelling features
        features, embedding_indexes = self._spelling_features(word, 0)
        for i in range(0,2+1):
            pos_feature = self._get_pos_features(pos_tag_sequence, position - i)
            features.extend(pos_feature)
            if i != 0:
                pos_feature = self._get_pos_features(pos_tag_sequence, position + i)
                features.extend(pos_feature)


        self._get_embedding_index(word.lower(), embedding_indexes)
        for i in range(1, 2 + 1):
            word_right = get_word(word_sequence, position + i)
            self._get_embedding_index(word_right.lower(), embedding_indexes)
            word_left = get_word(word_sequence, position - i)
            self._get_embedding_index(word_left.lower(), embedding_indexes)
        return features, embedding_indexes





    def _spelling_features(self, word, relative_position):
        """
        Extracts spelling features about the given word. Also, it considers the word's relative position.
        @type word: str
        @param word: given word to extract spelling features
        @type relative_position: int
        @param relative_position: relative position of word in the word sequence
        @return: a copy of the spelling_feature_cache for the specific (word, relative_position)
        """

        #features = dict()
        features = []

        features.append(int(is_capitalized(word)))
        features.append(int(word.isupper()))
        features.append(int(is_all_nonalphanumeric(word)))
        features.append(int(is_float(word)))
        # character_length
        features.append(float(1) / len(word))

        embedding_indexes = []
        for length in range(1, 5):
            # get prefix of word
            prefix = get_prefix(word, length).lower()
            self._get_embedding_index(prefix, embedding_indexes)

            suffix = get_suffix(word, length).lower()
            self._get_embedding_index(suffix, embedding_indexes)


        return features, embedding_indexes

    def _get_embedding_index(self, word, indexes):

        if word in self._word_embeddings:
            if word not in self._map_word_index_embedding:
                self._map_word_index_embedding[word] = self._embedding_index_counter
                self._embedding_index_counter += 1
            indexes.append(self._map_word_index_embedding[word])
        elif word.lower() in self._word_embeddings:
            if word not in self._map_word_index_embedding:
                self._map_word_index_embedding[word.lower()] = self._embedding_index_counter
                self._embedding_index_counter += 1
            indexes.append(self._map_word_index_embedding[word.lower()])
        else:
            random_emb = np.random.random(self.embedding_size)
            random_emb /= np.linalg.norm(random_emb)
            self._word_embeddings[word] = random_emb
            if word not in self._map_word_index_embedding:
                self._map_word_index_embedding[word] = self._embedding_index_counter
                self._embedding_index_counter += 1
            indexes.append(self._map_word_index_embedding[word])




    def _get_pos_features(self, pos_tag_sequence, position):
        pos_features = [0] * (len(self.pos_tag_list) + 1) # + 1 for any unknown pos tag
        pos = get_pos(pos_tag_sequence, position)
        if pos in self.pos_tag_list:
            pos_features[self.pos_tag_list[pos]] = 1
        else:
            pos_features[-1] = 1
        return pos_features

    def is_training(self):
        return self.is_training

