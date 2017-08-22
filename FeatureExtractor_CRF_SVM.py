import json
import os
import pickle

import fasttext
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

    def extract_features_crf(self, tokens_list, label_list, pos_list, extract_all):
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
                        self._get_features(word_sequence, position, pos_tag=pos_tags, numeric_feature=False))
                    # append location in locations list
                    sentence_locations[position] = ((sequence_num, position))

            self.length[sequence_num] = (int(len(word_sequence)))
            self.features_list[sequence_num] = (sentence_features)
            self.label_list[sequence_num] = (sentence_labels)
            self.location_list[sequence_num] = (sentence_locations)

        return self.features_list, self.label_list, self.length

    def load_word_embeddings(self, embedding_path, embedding_length, vocabulary):
        """
        Loads word embeddings from a file in the given path

        @type embedding_path: str
        @param embedding_path: path to the file containing the word embeddings
        """

        # load the word embeddings dictionary
        print("Loading word embeddings...")
        if "glove" in embedding_path:
            self.embedding_type = "glove"
            print("Load glove...")
            file_name = "word_embeddings_dict_" + str(embedding_length) + ".p"
            file_name = os.path.join(embedding_path, self.language, file_name)
            self._word_embeddings = pickle.load(open(file_name, "rb"))

            # the token for unknown word types must be present
            #       assert (
            #           self.unknown_symbol in self.__word_embeddings), "The token for unknown word types must be present in the embeddings file"

            # address some treebank token conventions.
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
        else:
            print("Loading fasttext ...")
            self.embedding_type = "fasttext"
            embedding_path = os.path.join(embedding_path, self.language, "wiki.{0}.bin".format(self.language))
            print(os.path.join(embedding_path))
            fasttext_script = os.path.join("fastText", "fasttext")
            vocabulary = set(map(lambda r: r.lower(), vocabulary))
            with open("tmp.txt", "w", encoding="utf-8") as file:
                file.write(" ".join(vocabulary))
            self._word_embeddings = dict()
            shell_command = '{0} print-word-vectors {1} < {2}'.format(fasttext_script,embedding_path, "tmp.txt")
            output = subprocess.check_output(shell_command, shell=True)
            for voc, emb in zip(vocabulary, output.decode().split("\n")):
                self._word_embeddings[voc] = list(
                    map(float, emb.split()[-self.embedding_size:]))


                #self._word_embeddings = os.path.join(embedding_path, self.language, "wiki.en.bin")
            #print(self._word_embeddings)
            #subprocess.call("pwd", shell=True)
            #subprocess.call("ls "+self._word_embeddings, shell=True)

            #subprocess.call("screen -dmLS fasttext bash -c \"fastText/fasttext print-word-vectors ../../word_embeddings/fasttext/en/wiki.en.bin;exec bash\"", shell=True)

    def is_training(self):
        return self.is_training

    # ====================================
    # NOT USED ANY MORE BUT KEEPED
    # ===================================




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
            self.max_length = max(self.max_length, length)
            self.label_list[sequence_num] = (label_sequence)
        self.feature_dim = max(self._map_feature_num2str.keys())
        total_label = len(set([item for sublist in self.label_list for item in sublist]))
        return self.max_length, total_label, self.feature_dim

    def transform_feature(self, data_to_transform, labels):
        features_tensorflow = np.zeros([0, self.feature_dim])
        for i, sentence in enumerate(data_to_transform):
            for j, feature in enumerate(sentence):
                vector = np.zeros([1, self.feature_dim])
                for k, (key, value) in enumerate(feature.items()):
                    vector[0, key - 1] = value
                features_tensorflow = np.append(features_tensorflow, vector, axis=0)

        output_label = []
        for sentence in labels:
            for label in sentence:
                output_label.append(label)

        return features_tensorflow, output_label

    def transform_feature_by_sentence(self, data_to_transform, label, nbr_classes):
        num_data = len(data_to_transform)

        features_tensorflow = np.zeros([num_data, self.max_length, self.feature_dim])
        for i, sentence in enumerate(data_to_transform):
            for j, feature in enumerate(sentence):
                for k, (key, value) in enumerate(feature.items()):
                    features_tensorflow[i][j][key - 1] = value
                    # for _ in range(max_length - len(sentence)):
                    #    features_tensorflow = np.append(features_tensorflow[i], np.zeros(max_value),axis =0)
        label_output = np.zeros([num_data, self.max_length]).astype(int)
        for i in range(len(label)):
            label_output[i] = np.append(label[i], [nbr_classes - 1] * (
                self.max_length - len(label[i])))
        return features_tensorflow, label_output

    def transform_sparce_feature(self, data_to_transform, label, nbr_classes):
        num_data = len(data_to_transform)

        indices = []
        values = []
        id = 0
        for i, sentence in enumerate(data_to_transform):
            for j, feature in enumerate(sentence):
                for k, (key, value) in enumerate(feature.items()):
                    indices.append((id, k))
                    values.append(value)
            id += 1
        # sparce_tensor = SparseTensor(indices=indices, values=values,
        #                             dense_shape=[len(data_to_transform) * self.max_length, self.feature_dim])
        label_output = np.zeros([num_data, self.max_length]).astype(int)
        for i in range(len(label)):
            label_output[i] = np.append(label[i],
                                        [nbr_classes] * (self.max_length - len(label[i])))
        print(np.shape(indices))
        print(np.shape(values))
        return indices, values, label_output

    def save_json_format(self, file_path):
        dataDict = {}
        dataDict['language'] = self.language
        dataDict['template'] = self.feature_template
        dataDict['featureStr2Index'] = self._map_feature_str2num
        dataDict['labelIndex2Str'] = self._map_label_num2str
        print('sanity check :')
        print('featureIndex2Str  size {0}'.format(len(self._map_feature_num2str)))
        print('labelIndex2Str  size {0}'.format(len(self._map_label_num2str)))
        with open(file_path, 'w') as outfile:
            json.dump(dataDict, outfile)

        print("json version stored at: {0}".format(file_path))
