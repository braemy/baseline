import math
import subprocess

#import fasttext
import numpy as np

from helper.utils import *
from helper.utils_data import *
from polyglot.mapping import Embedding
from polyglot.mapping import CaseExpander, DigitExpander


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
        self.embedding_type = None
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

    def load_word_embeddings(self,parameters, vocabulary):
        """
        Load word embedding for the given vocabulray
        embedding_type should be set in parameters (fasttext, fasttext_noOOV, polyglot, glove)
        embedding_size should be set in parameters according to the size of the given embedding_type
        embedding_path should be given in the parameters. Path to the main folder that contains all the embedding (without the embeding_type in the name)
        :parameter: parameters
        :vocabulary: set containing the vocab of the data
        """
        self.embedding_type = parameters['embedding_type']
        embedding_path = parameters['embedding_path']
        embedding_size = parameters['embedding_size']
        found_original_form = 0
        found_lower_case = 0
        not_found = 0
        number_of_OOV = 0
        # load the word embeddings dictionary
        print("Loading word embeddings...")
        if self.embedding_type == "glove":
            embedding_path = os.path.join(embedding_path, "glove")
            print("Load glove...")
            file_name = "word_embeddings_dict_" + str(embedding_size) + ".p"
            file_name = os.path.join(embedding_path, self.language, file_name)
            self._word_embeddings = pickle.load(open(file_name, "rb"))

        elif "fasttext" in self.embedding_type:
            print("Loading fasttext ...")
            path = os.path.join(embedding_path, "fasttext")
            token_list = []
            vocab_file_emb =  os.path.join(path, self.language, "vocab_word_embeddings_300.p")
            self.vocab_emb = load_pickle(vocab_file_emb, "utf-8")
            for v in vocabulary:
                if v in self.vocab_emb:
                    token_list.append(v)
                    found_original_form += 1
                elif v.lower() in self.vocab_emb:
                    token_list.append(v)
                    found_lower_case += 1
                elif self.embedding_type == "fasttext":
                    token_list.append(v)
                    number_of_OOV += 1
                elif self.embedding_type == "fasttext_noOOV":
                    not_found += 1
            print("Vocab size: {}".format(len(vocabulary)))
            print("Number of word found with original form {}".format(found_original_form))
            print("Number of word found with lowercase {}".format(found_lower_case))
            print("Number of OOV {}".format(number_of_OOV))
            print("Number of word not found {}".format(not_found))

            embedding_path = os.path.join(path, self.language, "wiki.{0}.bin".format(self.language))
            print(os.path.join(embedding_path))
            fasttext_script = os.path.join("fastText", "fasttext")
            with open("tmp.txt", "w", encoding="utf-8") as file:
                file.write(" ".join(token_list))
            self._word_embeddings = dict()
            shell_command = '{0} print-word-vectors {1} < {2}'.format(fasttext_script,embedding_path, "tmp.txt")
            output = subprocess.check_output(shell_command, shell=True)
            for voc, emb in zip(token_list, output.decode().split("\n")):
                self._word_embeddings[voc] = list(
                    map(float, emb.split()[-self.embedding_size:]))
        elif self.embedding_type == 'polyglot':
            embedding_path = os.path.join(embedding_path, "polyglot", self.language,"polyglot-{0}.pkl".format(self.language))
            vocab, embedding = load_pickle(embedding_path, encoding='latin1')
            self._word_embeddings = dict()
            for v, e in zip(vocab, embedding):
                self._word_embeddings[v] = e
        else:
            raise("Only support glove, fasttext, polyglot. Embedding tyoe {} not implemented".format(self.embedding_type))

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
            # if word.isdigit():
                # does the word contain only digit? 12334
            #    features["o_d({0})".format(relative_position)] = int(word.isdigit())
            # if has_numbers(word):
                # does the word contains at least one digit? joe
            #    features["a_d({0})".format(relative_position)] = int(has_numbers(word))
            if is_all_nonalphanumeric(word):
                # check if all chars are nonalphanumeric
                features["nonalpha({0})".format(relative_position)] = int(is_all_nonalphanumeric(word))
            if is_float(word):
                # check if word can be converted to float, i.e. word is a number
                features["f({0})".format(relative_position)] = int(is_float(word))
            # character_length
            # features["c_l({0})".format(relative_position)] = float(1) / len(word)

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
            elif self.morphological_features == "embedding":
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
                features["w({0})={1}".format(position, word)] = 1
            elif self.morphological_features == "embedding":
                self._get_word_embeddings(word_sequence, position, 0, features, "w")
                self._get_word_embeddings(word_sequence, position, 0, features, "w_l")
            else:
                raise ("Unrecognized feature type")
        # get 2 words on the left and right¨
        # add features for the words on the left and right side
        for i in range(1, 2 + 1):
            word_right = get_word(word_sequence, position + i)
            word_left = get_word(word_sequence, position - i)
            if self.morphological_features == "regular":
                # words around the word to predict
                features["w({0})={1}".format(i, word_right)] = 1
                features["w({0})={1}".format(-i, word_left)] = 1
            elif self.morphological_features == "embedding":
                self._get_word_embeddings(word_sequence, position, i, features, "w")
                self._get_word_embeddings(word_sequence, position, -i, features, "w")
            else:
                raise ("Unrecognized feature type")

        if pos_tag_sequence is not None:
            features.update(self._get_pos_features(pos_tag_sequence=pos_tag_sequence, position=position))

        return features

    def _get_pos_features(self, pos_tag_sequence, position):
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

    def _get_word_embeddings(self, word_sequence, position, offset, features, name="wb"):
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
            if self.embedding_type in ["glove", "polyglot", "fasttext_noOOV"]:
                word_embedding = self._word_embeddings.get(word, np.random.rand(self.embedding_size))
            elif self.embedding_type == "fasttext":
                word_embedding = self._word_embeddings[word]
            else:
                raise("Only support glove, polyglot, fasttext")

            word_embedding /= np.linalg.norm(word_embedding)
        offset = str(offset) if offset <= 0 else "+" + str(offset)
        for i in range(len(word_embedding)):
            if word_embedding[i] != 0:
                features[name + "({0})_at({1})".format(offset, (i + 1))] = word_embedding[i]

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
            self._get_word_embeddings(word_sequence, position, i, features)
        return features

    def reset(self):
        """
        reset the features parameters
        used for cross validation
        :return:
        """
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

    def is_training(self):
        """
        return if the feature extrator is training or testing
        is_training should be False when it extract features of the test set
        :return: value of is_training
        """
        return self.is_training

