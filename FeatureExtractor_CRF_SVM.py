# -*- coding: utf-8 -*-
import json
import os
import pickle
import subprocess

import numpy as np
from tqdm import tqdm

from helper.FeatureExtractor import FeatureExtractor


class FeatureExtractor_CRF_SVM(FeatureExtractor):
    """
    Extracts features from sequence data
    """

    def __init__(self, feature_template, language, embedding_size=None):
        FeatureExtractor.__init__(self, feature_template, language
                                  , embedding_size)
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
        # symbol for unknown word that must be contained in the embeddings and bitstrings dictionaries
        self.unknown_symbol = "<?>"
        self.features_list = []
        self.label_list = []
        self.location_list = []
        self.max_length = -1
        if embedding_size is not None:
            self.embedding_size = int(embedding_size)
            self.embedding_start = np.zeros(self.embedding_size)
            self.embedding_end = np.ones(self.embedding_size) / np.linalg.norm(np.ones(self.embedding_size))

        self.block_size = 50000
        self.quiet = False
        self.embedding_type = None

    def extract_features_svm(self, sequence_data, extract_all):
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

        # list for labels
        label_list = []
        # list for features
        self.features_list = []
        # list for locations
        location_list = []
        # extract data path from sequence_data
        self.data_path = sequence_data.data_path
        # iterate through all sequences (=sentences) and all words in each sentence
        for sequence_num, (word_sequence, *label_pos_sequence) in tqdm(enumerate(sequence_data.sequence_pairs)):
            # check if relational features are used
            # if so, build relational info for the current word_sequence
            label_sequence = label_pos_sequence[0]
            pos_sequence = None if len(label_pos_sequence) == 1 else label_pos_sequence[1]
            pos_sequence = None
            for position, label in enumerate(label_sequence):
                # only use labeled instances unless extract_all=True.
                if (label is not None) or extract_all:
                    # append label id to label list
                    label_list.append(self._get_label(label))
                    # append feature id in features list
                    self.features_list.append(
                        self._get_features(word_sequence, position, pos_sequence, numeric_feature=True))
                    # append location in locations list
                    location_list.append((sequence_num, position))
        return self.features_list, label_list, location_list

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
        # list for locations
        self.location_list = [None] * len(tokens_list)
        # iterate through all sequences (=sentences) and all words in each sentence
        for sequence_num, (word_sequence, labels_sequence, pos_sequence) in tqdm(
                enumerate(zip(tokens_list, label_list, pos_list)), "Feature extraction", miniters=200, mininterval=2,
                disable=self.quiet):

            sentence_features = [None] * len(labels_sequence)
            sentence_labels = [None] * len(labels_sequence)
            sentence_locations = [None] * len(labels_sequence)

            for position, (label, pos_tags) in enumerate(zip(labels_sequence, pos_sequence)):

                # only use labeled instances unless extract_all=True.
                if (label is not None) or extract_all:
                    # append label id to label list
                    sentence_labels[position] = (self._get_label(label))
                    # append feature id in features list
                    sentence_features[position] = (
                        self._get_features(word_sequence, position, pos_tag=pos_sequence, numeric_feature=numeric_feature))
                    # append location in locations list
                    sentence_locations[position] = ((sequence_num, position))

            self.length[sequence_num] = (int(len(word_sequence)))
            self.features_list[sequence_num] = (sentence_features)
            self.label_list[sequence_num] = (sentence_labels)
            self.location_list[sequence_num] = (sentence_locations)

        return self.features_list, self.label_list, self.length


    def is_training(self):
        return self.is_training
