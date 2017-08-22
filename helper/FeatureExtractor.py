import math
import subprocess

import fasttext
import numpy as np

from helper.utils import *
from helper.utils_data import *


class FeatureExtractor(object):
    """
	Extracts features from sequence data
	"""

    def __init__(self, feature_template, language, embedding_size=None):
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
        # dictionary that maps label string to number
        self._map_label_str2num = {}
        # dictionary that maps label number to string
        self._map_label_num2str = {}
        # dictionary that contains word embeddings (key = word, value = float array)
        self._word_embeddings = None
        # symbol for unknown word that must be contained in the embeddings and bitstrings dictionaries
        self.unknown_symbol = "<?>"
        self.features_list = []
        self.label_list = []
        self.location_list = []
        self.max_sentence_length = -1
        if embedding_size is not None:
            self.embedding_size = int(embedding_size)
            self.embedding_start = np.zeros(self.embedding_size)
            self.embedding_end = np.ones(self.embedding_size) / np.linalg.norm(np.ones(self.embedding_size))

        self.block_size = 50000
        self.token_features2 = True
        self.morphological_features = "regular"
        self.token_features1 = True
        self.token_features0 = True

        self.window_size = 1

    def get_feature_dim(self):
        return len(self._map_feature_str2num.keys())

    def _get_label(self, label):
        """
        Return the label in hot encoding format

        @type label: str
        @param label: label for which the label id is required
        @return: integer ID for given label
        """

        if self.is_training:
            # if training, add unknown label types to the dictionary
            if label not in self._map_label_str2num:
                # each time a new label arrives, the counter is incremented (start indexing from 0)
                label_number = len(self._map_label_str2num)
                # make the label string <--> id mapping in the dictionary
                self._map_label_str2num[label] = label_number
                self._map_label_num2str[label_number] = label
            # return label id
            return self._map_label_str2num[label]
        else:
            # if predicting, take value from the trained dictionary
            if label in self._map_label_str2num:
                return self._map_label_str2num[label]
            # if label is not found, return -1
            else:
                return -1

    def num_feature_types(self):
        """
		Finds the number of distinct feature types

		@return: the number of distinct feature types
		"""

        return len(self._map_feature_str2num)

    def get_feature_string(self, feature_number):
        """
		Converts a numeric feature ID to a string

		@type feature_number: int
		@param feature_number: numeric id of feature
		@return: given a feature number, it returns the respective feature string
		"""

        assert (feature_number in self._map_feature_num2str), "Feature id not in featureID-to-string dictionary"
        return self._map_feature_num2str[feature_number]

    def get_label_string(self, label_number):
        """
		Converts a numeric label ID to a string

		@type label_number: int
		@param label_number: numeric id of id
		@return: the label string that corresponds to the given label number
		"""

        assert (label_number in self._map_label_num2str), "Label id not in labelID-to-string dictionary"
        # if label_number not in self._map_label_num2str:
        #    return "O"
        return self._map_label_num2str[label_number]

    def get_feature_number(self, feature_string):
        """
		Converts a feature string to a numeric ID

		@type feature_string: str
		@param feature_string: feature in string format
		@return: the numeric feature id given the feature string
		"""

        assert (feature_string in self._map_feature_str2num), "Feature string not in featureString-to-ID dictionary"
        return self._map_feature_str2num[feature_string]

    def get_label_number(self, label_string):
        """
		Converts a label string to a numeric ID

		@type label_string: str
		@param label_string: label in string format
		@return: the numeric label id given the label string
		"""

        assert (label_string in self._map_label_str2num), "Label string not in labelString-to-ID dictionary"
        # if label_string not in self._map_feature_str2num:
        #    return self._map_feature_str2num["O"]
        return self._map_label_str2num[label_string]

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

        position = position
        # Extract raw features depending on the given feature template
        if self.feature_template == "baseline":
            raw_features = self._get_baseline_features(word_sequence, position, pos_tag)
        elif self.feature_template == "embedding":
            assert (self._word_embeddings is not None), "A path to embedding file should be given"
            raw_features = self._get_embedding_features(word_sequence, position, pos_tag)
        else:
            raise Exception("Unsupported feature template {0}".format(self.feature_template))
        # map extracted raw features to numeric
        numeric_features = self._map_raw_to_numeric_features(raw_features)
        if numeric_feature:
            return numeric_features
        else:
            return raw_features

    def _map_raw_to_numeric_features(self, raw_features):
        """
            Maps raw features to numeric
    
            @type raw_features: dict
            @param raw_features: dictionary of raw features (key = feature string, value = feature value)
            @return: a numeric dictionary (key = feature id, value = feature value)
        """

        numeric_features = {}
        # iterate through all given features
        for raw_feature in raw_features:
            if self.is_training:
                # if training, add unknown feature types to the dictionary
                if raw_feature not in self._map_feature_str2num:
                    # fix feature id for a given string feature
                    # Note: Feature index has to starts from 1 in liblinear
                    feature_number = len(self._map_feature_str2num) + 1
                    # do the mapping for the feature string <--> id
                    self._map_feature_str2num[raw_feature] = feature_number
                    self._map_feature_num2str[feature_number] = raw_feature
                # set value of the key=feature_id to the correct feature value from the raw_features dict
                numeric_features[self._map_feature_str2num[raw_feature]] = raw_features[raw_feature]
            else:
                # if predicting, only consider known feature types.
                if raw_feature in self._map_feature_str2num:
                    numeric_features[self._map_feature_str2num[raw_feature]] = raw_features[raw_feature]
        return numeric_features

    def load_word_embeddings(self, embedding_path, embedding_length):
        """
        Loads word embeddings from a file in the given path

        @type embedding_path: str
        @param embedding_path: path to the file containing the word embeddings
        """

        # load the word embeddings dictionary
        if "glove" in embedding_path:
            print("Loading word embeddings...")
            file_name = "word_embeddings_dict_" + str(embedding_length) + ".p"
            file_name = os.path.join(embedding_path, self.language, file_name)
            with open(file_name, "rb") as file:
                self._word_embeddings = pickle.load(file)
        if "fasttext" in embedding_path:
            print("YOu should have start a screen with fasttext")
            print("Loading FastText word embeddings...")
            # fasttext supports only word embeddings with 300 dimensionality
            assert (embedding_length == 300), "Embedding length should be 300 when using FastText"
            file_name = os.path.join(embedding_path, self.language, "wiki.en.bin")
            # load fasttext word embeddings
            print(file_name)
            self._word_embeddings = fasttext.load_model(file_name)
            # the token for unknown word types must be present
            # assert (self.unknown_symbol in self.__word_embeddings), "The <?> must be present in the embeddings file"
            #       assert self._word_embeddings is not None, "Use glove or FastText embeddings"
        print("Words embeddings loaded")

        # address some treebank token conventions.
        if "glove" in embedding_path:
            if "(" in self._word_embeddings:
                self._word_embeddings["-LCB-"] = self._word_embeddings["("]
                self._word_embeddings["-LRB-"] = self._word_embeddings["("]
                self._word_embeddings["*LCB*"] = self._word_embeddings["("]
                self._word_embeddings["*LRB*"] = self._word_embeddings["("]
            if ")" in self._word_embeddings:
                self._word_embeddings["-RCB-"] = self._word_embeddings[")"]
                self._word_embeddings["-RRB-"] = self._word_embeddings[")"]
                self._word_embeddings["*RCB*"] = self._word_embeddings[")"]
                self._word_embeddings["*RRB*"] = self._word_embeddings[")"]
            if "\"" in self._word_embeddings:
                self._word_embeddings["``"] = self._word_embeddings["\""]
                self._word_embeddings["''"] = self._word_embeddings["\""]
                self._word_embeddings["`"] = self._word_embeddings["\""]
                self._word_embeddings["'"] = self._word_embeddings["\""]

    def _spelling_features(self, word, relative_position):
        """
		Extracts spelling features about the given word. Also, it considers the word's relative position.

		@type word: str
		@param word: given word to extract spelling features
		@type relative_position: int
		@param relative_position: relative position of word in the word sequence
		@return: a copy of the spelling_feature_cache for the specific (word, relative_position)
		"""

        if (word, relative_position) not in self.spelling_feature_cache:
            features = dict()
            # check if word is capitalized
            # features["is_capitalized({0})={1}".format(relative_position, is_capitalized(word))] = 1
            # features["is_upper({0})={1}".format(relative_position, word.isupper())] = 1
            # features["contains_only_digit({0})={1}".format(relative_position, word.isdigit())] = 1
            # features["contains_any_digit({0})={1}".format(relative_position, has_numbers(word))] = 1
            # features["character_length({0})={1}".format(relative_position, len(word))] = 1
            # check if all chars are nonalphanumeric
            # features["is_all_nonalphanumeric({0})={1}".format(relative_position, is_all_nonalphanumeric(word))] = 1
            # check if word can be converted to float, i.e. word is a number
            # features["is_float({0})={1}".format(relative_position, is_float(word))] = 1

            if is_capitalized(word):
                # is the word capialized? Joe
                features["cap({0})".format(relative_position)] = int(is_capitalized(word))
            if word.isupper():
                # is the word all in upper case? HELLO
                features["up({0})".format(relative_position)] = int(word.isupper())
            if word.isdigit():
                # does the word contain only digit? 12334
                features["o_d({0})".format(relative_position)] = int(word.isdigit())
            if has_numbers(word):
                # does the word contains at least one digit? joe
                features["a_d({0})".format(relative_position)] = int(has_numbers(word))
            if is_all_nonalphanumeric(word):
                # check if all chars are nonalphanumeric
                features["non_alpa({0})".format(relative_position)] = int(is_all_nonalphanumeric(word))
            if is_float(word):
                # check if word can be converted to float, i.e. word is a number
                features["f({0})".format(relative_position)] = int(is_float(word))
            # character_length
            features["c_l({0})".format(relative_position)] = float(1) / len(word)

            # build suffixes and preffixes for each word (up to a length of 4)
            if self.morphological_features == "regular":
                for length in range(1, 5):
                    features[
                        # prefix
                        "p{0}({1})={2}".format(length, relative_position, get_prefix(word, length))] = 1
                    features[
                        # suffix
                        "s{0}({1})={2}".format(length, relative_position, get_suffix(word, length))] = 1



            # use fasttext to create morphological features
            elif self.morphological_features == "embeddings":
                for length in range(1, 5):
                    # get prefix of word
                    prefix = get_prefix(word, length)

                    if prefix in self._word_embeddings:
                        # get the word embeddings for the prefix
                        prefix_embedding = self._word_embeddings[prefix.lower()]
                        # normalize embeddings
                        prefix_embedding /= np.linalg.norm(prefix_embedding)
                        # enrich given features dict
                        for i, value in enumerate(prefix_embedding):
                            if value != 0:
                                features["ngram{0}({1})_at({2})".format(length, relative_position, (i + 1))] = value
                    else:
                        # initialize embeddings to zero
                        # prefix_embedding = np.zeros((self.embedding_size))
                        # create a unique numpy array for each prefix of a specific length
                        # prefix_embedding[length - 1] = 1
                        # normalize embeddings
                        # prefix_embedding /= np.linalg.norm(prefix_embedding)
                        # enrich given features dict

                        # for i, value in enumerate(prefix_embedding):
                        #    features["ngram{0}({1})_at({2})".format(length, relative_position, (i + 1))] = value

                        features["ngram{0}({1})_at({2})".format(length, relative_position, length - 1)] = 1
            else:
                raise ("Unrecognized feature type")

            self.spelling_feature_cache[(word, relative_position)] = features

        # Return a copy so that modifying that object doesn't modify the cache.
        return self.spelling_feature_cache[(word, relative_position)].copy()

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
        features = {}

        # get word at given position
        word = get_word(word_sequence, position)

        # extract spelling features
        if self.morphological_features is not None:
            features.update(self._spelling_features(word, 0))

        # identify word
        if self.token_features0:
            if self.morphological_features == "regular":
                # the word in lower case
                features["w_l({0})={1}".format(position, word.lower())] = 1
                # the word like it appears in the text
                features["w({0})={1}".format(position, word)] = 1
            elif self.morphological_features == "embeddings":
                self.__get_word_embeddings(word_sequence, position, 0, features, "w")
                self.__get_word_embeddings(word_sequence, position, 0, features, "w_l")
            else:
                raise ("Unrecognized feature type")
        # get 2 words on the left and right¨
        # add features for the words on the left and right side
        for i in range(1, 4 + 1):
            word_right = get_word(word_sequence, position + i)
            word_left = get_word(word_sequence, position - i)
            if self.morphological_features == "regular":
                # words around the word to predict
                features["w({0})={1}".format(i, word_right)] = 1
                features["w({0})={1}".format(-i, word_left)] = 1
            elif self.morphological_features == "embeddings":
                self.__get_word_embeddings(word_sequence, position, i, features, "w")
                self.__get_word_embeddings(word_sequence, position, -i, features, "w")
            else:
                raise ("Unrecognized feature type")

            features.update(self._spelling_features(word_right, i))
            features.update(self._spelling_features(word_left, -i))

        # if pos_tag_sequence:

        if pos_tag_sequence is not None:
            features.update(self.__get_pos_features(pos_tag_sequence=pos_tag_sequence, position=position))

        return features

    def __get_pos_features(self, pos_tag_sequence, position):
        """
        Build the pos tag features
        :param word_sequence:
        :param position:
        :param pos_tag_sequence:
        :return:
        """
        pos = get_pos(pos_tag_sequence, position)
        pos_left1 = get_pos(pos_tag_sequence, position - 1)
        pos_left2 = get_pos(pos_tag_sequence, position - 2)
        pos_right1 = get_pos(pos_tag_sequence, position + 1)
        pos_right2 = get_pos(pos_tag_sequence, position + 2)

        features = dict()

        features["pos(-1) = {0}".format(pos_left1)] = 1
        features["pos(-2) = {0}".format(pos_left2)] = 1
        features["pos(0) = {0}".format(pos)] = 1
        features["pos(+1) = {0}".format(pos_right1)] = 1
        features["pos(+2) = {0}".format(pos_right2)] = 1

        return features

    def __get_word_embeddings(self, word_sequence, position, offset, features, name="wb"):
        """
		Gets embeddings for a given word using the embeddings dictionary

		@type word_sequence: list
		@param word_sequence: sequence of words
		@type position: int
		@param position: position in the sequence of words
		@type offset: int
		@param offset: offset relative to position
		@type features: dict
		@param features: dictionary with features
		"""

        # get current word
        if position + offset < 0:
            # before the sentence
            word_embedding = self.embedding_start
        elif (position + offset) > len(word_sequence) - 1:
            # after the sequence
            word_embedding = self.embedding_end

        else:
            word = word_sequence[position + offset]
            word = word.lower()
            if self.embedding_type == "glove":
                word_embedding = self._word_embeddings.get(word, np.random.rand(self.embedding_size))
            else:
                word_embedding = self._word_embeddings[word]

            word_embedding /= np.linalg.norm(word_embedding)
        offset = str(offset) if offset <= 0 else "+" + str(offset)
        for i in range(len(word_embedding)):
            if word_embedding[i] != 0:
                features[name + "({0})_at({1})".format(offset, (i + 1))] = word_embedding[i]
                # features.update(dict(enumerate(word_embedding)))

    def _get_embedding_features(self, word_sequence, position, pos_tags):
        """
		Extract embedding features = normalized baseline features + (normalized) embeddings
		of current, left, and right words.

		@type word_sequence: list
		@param word_sequence: sequence of words
		@type position: int
		@param position: position in the sequence of words
		@return: full dict of features
		"""
        # compute the baseline feature vector and normalize its length to 1
        features = self._get_baseline_features(word_sequence, position, pos_tags)
        # assumes binary feature values
        norm_features = math.sqrt(len(features))
        # normalize
        for feature in features:
            features[feature] /= norm_features
            # extract word embedding for given and neighbor words
        for i in range(-self.window_size, self.window_size + 1):
            self.__get_word_embeddings(word_sequence, position, i, features)
        # if position > 0:
        #    self.__get_word_embeddings(word_sequence, position, -1, features)
        # if position < len(word_sequence) - 1:
        #    self.__get_word_embeddings(word_sequence, position, 1, features)

        return features

    def reset(self):
        # dictionary with all spelling features
        self.spelling_feature_cache = {}
        # path to data
        self.data_path = None
        # boolean flag for training
        self.is_training = True
        # dictionary that maps feature string to number
        self._map_feature_str2num = {}
        # dictionary that maps feature number to string
        self._map_feature_num2str = {}
        # dictionary that maps label string to number
        self._map_label_str2num = {}
        # dictionary that maps label number to string
        self._map_label_num2str = {}

        # symbol for unknown word that must be contained in the embeddings and bitstrings dictionaries
        self.unknown_symbol = "<?>"
        self.features_list = []
        self.label_list = []
        self.location_list = []
        self.max_sentence_length = -1

    def save_preprocessing(self, dir):
        # remove unused data
        del self._word_embeddings
        del self.spelling_feature_cache
        # save the features
        self.__save_training_features(dir)
        self.pickle_file(dir, "validation_features")
        self.pickle_file(dir, "test_features")
        self.pickle_file(dir, "_map_label_str2num")
        self.pickle_file(dir, "_map_label_num2str")
        self.pickle_file(dir, "_map_feature_str2num")
        self.pickle_file(dir, "_map_feature_num2str")
        # save the features
        # self.__save_training_features(dir)
        # self.pickle_file(dir, "map_label2id")
        # self.pickle_file(dir, "map_id2label")
        # self.pickle_file(dir, "map_feature_str2num")
        # self.pickle_file(dir, "map_feature_num2str")

    def __save_training_features(self, dir):
        for block_num, i in enumerate(range(0, len(self.train_features), self.block_size)):
            with open(os.path.join(dir, "train_features_" + str(block_num) + ".p"), "wb") as file:
                pickle.dump(self.train_features[i: i + self.block_size], file)

    def __load_features(self, dir, file_name):
        i = 0
        tmè = np.empty(0)
        while os.path.exists(os.path.join(dir, file_name + str(i) + ".p")):
            with open(os.path.join(dir, file_name + str(i) + ".p"), "rb") as file:
                tmp = np.append(tmp, pickle.load(file), axis=0)
            i += 1
        return tmp

    def load_preprocessing(self, dir):
        self.train_features = self.__load_features(dir, "train_features_")
        self.validation_features = self.load_file(dir, "validation_features")
        self.test_features = self.load_file(dir, "test_features")
        self._map_label_str2num = self.load_file(dir, "_map_label_num2str")
        self._map_label_str2num = self.load_file(dir, "_map_label_str2num")
        self._map_feature_str2num = self.load_file(dir, "_map_feature_str2num")
        self._map_feature_num2str = self.load_file(dir, "_map_feature_num2str")

        self.features_dim_train = len(self._map_feature_str2num.keys())
        self.num_classes_train = len(self._map_label_str2num.keys())
        self.num_tokens_train = len(self.train_features)
        self.features_dim_validation = len(self._map_feature_str2num.keys())
        self.num_classes_validation = len(self._map_label_str2num.keys())
        self.num_tokens_validation = len(self.validation_features)
        self.features_dim_test = len(self._map_feature_str2num.keys())
        self.num_classes_test = len(self._map_label_str2num.keys())
        self.num_tokens_test = len(self.test_features)

    def pickle_file(self, dir, name):
        with open(os.path.join(dir, name + ".p"), "wb") as file:
            pickle.dump(getattr(self, name), file)

    def load_file(self, dir, name):
        with open(os.path.join(dir, name + ".p"), "rb") as file:
            return pickle.load(file)

    def is_training(self):
        return self.is_training

    def display_summary(self):

        print("Feature template:", "\"" + self.feature_template + "\"")
        print("Number of sentences/tokens in train:", len(self.train_features))
        print("Number of sentences/tokens in validation:", len(self.validation_features))
        print("Number of sentences/tokens: in test", len(self.test_features))

    def build_feature_map(self, sequence_data, extract_all):
        self.label_list = [None] * len(sequence_data.sequence_pairs)

        for sequence_num, (word_sequence, *label_pos_sequence) in enumerate(sequence_data.sequence_pairs):
            label_sequence = label_pos_sequence[0]
            pos_sequence = None if len(label_pos_sequence) == 1 else label_pos_sequence[1]

            if sequence_num % 10000 == 0:
                print("Extracting features:", sequence_num)
            for position, label in enumerate(label_sequence):

                # only use labeled instances unless extract_all=True.
                if (label is not None) or extract_all:
                    # append label id to label list
                    self._get_label(label)
                    # append feature id in features list
                    self._get_features(word_sequence, position, pos_sequence, numeric_feature=True)
                    # append location in locations list

            length = len(word_sequence)
            self.max_sentence_length = max(self.max_sentence_length, length)
            self.label_list[sequence_num] = (label_sequence)
        self.feature_dim = max(self._map_feature_num2str.keys())
        total_label = len(set([item for sublist in self.label_list for item in sublist]))
        return self.max_sentence_length, total_label, self.feature_dim
