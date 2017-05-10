import collections
import numpy as np
import copy
import os
import pickle


class SequenceData(object):
    """
	Represents a data set of sequences of words with labels
	The sequences can be partially labeled
	The data be loaded from a text file or a list
	"""

    def __init__(self, given_data, pos_tag, convert_to_id=False):
        # path to the data
        self.data_path = None
        # number of words in the data
        self.num_of_words = 0
        # number of labeled words
        self.num_labeled_words = 0
        # boolean if the data set is partially labeled
        self.is_partially_labeled = False
        # contains sequence of: sentences (sequence of words) | sequence of labels
        self.sequence_pairs = []
        # dictionary for counting: key = label | value = number of occurrences for each label
        self.label_count = collections.Counter()
        # dictionary for counting: key = word | value = number of occurrences for each word
        self.word_count = collections.Counter()
        # dictionary for counting: key = word | value = dictionary for counting labels for each word
        # e.g. { "set" : {"verb":10, "noun":8} }
        # e.g. { "eat" : {"verb":15} }
        self.word_label_count = {}
        # added 21.02.17
        # dictionary for counting: key = word | value = dictionary for counting the POS tag for each word
        self.word_pos_count = {}
        # dictionary for counting: key = pos tag | value = number of occurrences for each pos tag
        self.pos_count = collections.Counter()

        self.convert_to_id = False

        # check if given data is a path to file, i.e. a string
        if isinstance(given_data, str):
            # initialize sequence pairs from file
            self.__initialize_sequence_pairs_from_file(given_data, pos_tag)
        # check if given data is in form of a list
        elif isinstance(given_data, list):
            # initialize sequence pairs from list
            self.__initialize_sequence_pairs_from_list(given_data)
        # through exception if neither a file, nor a list are provided for given_data
        else:
            raise Exception("SequenceData can be initialized using either a string (file path) or a list")

        self.__initialize_sequencedata_attributes()

    #def get_token_id(self):# TODO HEEEEEERE !!!!!!!!!!!!!!!!!!


    def merge_sequence(self, sequence_to_merge):
        self.sequence_pairs = np.append(self.sequence_pairs, sequence_to_merge.sequence_pairs, axis = 0) # TODO update all variable of self
        return self

    def get_subsequence_pairs(self, start, end):
        copy_sequence = copy.copy(self)
        if end is None:
            copy_sequence.sequence_pairs = copy_sequence.sequence_pairs[start:]
        else:
            copy_sequence.sequence_pairs = copy_sequence.sequence_pairs[start:end]
        return copy_sequence

    def split_in_2_sequences(self,start,end):
        test = self.get_subsequence_pairs(start, end)
        train1 = self.get_subsequence_pairs(0, start)
        train2 = self.get_subsequence_pairs(end,None)
        train1.sequence_pairs = np.append(train1.sequence_pairs, train2.sequence_pairs, axis=0)
        return train1, test

    def get_sequence_average_length(self):
        """
		Calculates the average length of the sequences

		@return: average length of sequences (sentences) in the given data set
		"""

        length_sum = 0
        for word_sequence, _ in self.sequence_pairs:
            length_sum += len(word_sequence)
        return float(length_sum) / len(self.sequence_pairs)

    def get_labels(self):
        labels = [""] * len(self.sequence_pairs)
        total_o = 0
        total_label = 0
        for seq_num, sequence in enumerate(self.sequence_pairs):
            labels[seq_num] = sequence[1]
            total_label += len(sequence[1])
            total_o += np.sum([s == 'O' for s in sequence[1]])
        percentage_o = round(total_o/total_label*100, 2) if total_label > 0 else 0
        return labels, percentage_o

    def split_train_validation(self, train_ratio):
        """
        randomize the sentences
        and split the sentences according to the ratio ratio
        :param train_ratio:
        :type train_ratio:
        :param dev_ratio:
        :type dev_ratio:
        :param test_ratio:
        :type test_ratio:
        :return:
        :rtype:
        """

        assert train_ratio <=1, " train ration should be less or equal to 1 "+train_ratio
        validation_ratio = round(1-train_ratio,2)
        combine_sequence = copy.copy(self)

        #randomize the sequence_pairs
        np.random.shuffle(combine_sequence.sequence_pairs)

        total_sequence = len(combine_sequence.sequence_pairs)
        train_sequence = combine_sequence.get_subsequence_pairs(start=0, end= int(train_ratio * total_sequence))
        dev_sequence = combine_sequence.get_subsequence_pairs(start=int(train_ratio * total_sequence), end= None)

        return train_sequence, dev_sequence

    def convert_sequence_to_token(self, num_tokens=None):
        """
        Convert a Sequence to a list of token and a list of label
        To create the features of one token, we still need the context of the word (position, word before and after),
        the token list keep track of the sentence number and position in the sentence
        :param data:
        :type data:
        :return:
        :rtype:
        """
        sequence_pairs =[]
        if num_tokens is None:
            token_list = []
            labels_list = []
            for i, d in enumerate(self.sequence_pairs):
                sentence = d[0]
                labels = d[1]
                sentence_tokens =[]

                for position, (token, label) in enumerate(zip(sentence, labels)):
                    token_list.append({'t':token, 'l':label, 'sent': i ,'position':position,'sent_token': str(i)+"_"+str(position)})
                    labels_list.append(label)

                    sentence_tokens.append({'t':token, 'l':label, 'sent': i ,'sent_token': str(i)+"_"+str(position)})
                sequence_pairs.append([sentence_tokens, labels])
            sequence_copy = copy.copy(self)
            sequence_copy.sequence_pairs = sequence_pairs

        else:
            #token_list = []
            #labels_list = []
            token_list = np.empty([num_tokens], dtype="<U6")
            labels_list = np.empty([num_tokens], dtype="<U6")
            counter = 0
            for i, d in enumerate(self.sequence_pairs):
                sentence = d[0]
                labels = d[1]
                for position, (token, label) in enumerate(zip(sentence,labels)):
                    #token_list.append({'t':token, 'l':label, 'sent':i, 'pos':position})
                    #labels_list.append(label)
                    token_list[counter]  = token
                    labels_list[counter] = label
                    counter += 1
        return token_list, labels_list, sequence_copy

    def split_token_label(self, num_of_sentences=50000):
        size = len(self.sequence_pairs) if len(self.sequence_pairs) < num_of_sentences else num_of_sentences
        list_tokens_sequence = [None] * size
        list_labels_sequence = [None] * size
        for sequence_num, (word_sequence, *label_pos_sequence) in enumerate(self.sequence_pairs):
            if sequence_num >= size:
                break
            list_tokens_sequence[sequence_num] = word_sequence
            list_labels_sequence[sequence_num] = label_pos_sequence[0]

        return list_tokens_sequence, list_labels_sequence

    def __load_vocab(self):
        file_name = "../../word_embeddings/en/vocab_word_embeddings_50.p"
        with open(file_name, "rb") as file:
            return pickle.load(file)

    def __initialize_sequence_pairs_from_file(self, data_path, pos_tag):
        """
		Initializes sequences from a text file
		The format of the file should be [word] [Part-of-Speech tagging (if present)] [true label] [predicted label]
		True and predicted labels are optional
		Empty lines indicate sequence boundaries

		@type data_path: str
		@param data_path: the path to the data
		@type pos_tag: boolean
		@param pos_tag: boolean that say if the files containt part of speech tagging or not

		"""
        self.data_path = data_path
        with open(data_path, "r") as input_file:
            word_sequence = []
            pos_sequence = []
            label_sequence = []
            id_token = []

            for line in input_file:
                tokens = line.split()
                # if not pos_tag:
                #    assert (len(tokens) < 3), "Each line should contain no more than 3 columns"
                # if pos_tag:
                #    assert (len(tokens) < 4), "Each line should contains no more than 4 columns"
                # if line is not empty
                if tokens:
                    # the word is the 1st token
                    word = tokens[0]
                    if pos_tag:
                        pos = tokens[1]
                    # the label is the 3td token, if it exists, otherwise label = None
                    if pos_tag:
                        label = None if len(tokens) == 2 else tokens[2] # TODO have a look for POS
                    else:
                        label = None if len(tokens) == 1 else tokens[1]
                    # else:
                    #    if len(tokens) ==
                    #    # the label is the 2nd token, if it exists, otherwise label = None
                    #    label = None if len(tokens) == 1 else tokens[1]

                    if label is None:
                        # set the partially labeled flag to True
                        self.is_partially_labeled = True
                    # build sequence of words
                    word_sequence.append(word)
                    # build sequence of labels
                    label_sequence.append(label)
                    if pos_tag:
                        # build_sequence of pos tags
                        pos_sequence.append(pos)
                # line is empty means that a new sentence if about to start
                else:
                    # check if word_sequence is empty
                    if word_sequence:
                        if pos_tag:
                            # append word_sequence and label_sequence and pos_sequence to sequence_pairs
                            self.sequence_pairs.append([word_sequence, label_sequence, pos_sequence])
                        else:
                            self.sequence_pairs.append([word_sequence, label_sequence])
                        # flush lists for the next sentence
                        word_sequence = []
                        label_sequence = []
                        pos_sequence = []

            # file may not end with empty line
            # data of last sentence should be also be appended to sequence_pairs
            if word_sequence:
                if pos_tag:
                    self.sequence_pairs.append([word_sequence, label_sequence, pos_sequence])
                else:
                    self.sequence_pairs.append([word_sequence, label_sequence])

    def __initialize_sequence_pairs_from_list(self, sequence_list):
        """
		Initializes sequences from a given list
		Absent labels are denoted with None

		@type sequence_list: list
		@param sequence_list: the list contains sequence of words with their
		respective labels. Each element of the given list should have the
		following form:
		sequence_list[i] = [word_sequence, label_sequence]
		"""

        for sequence_pair in sequence_list:
            assert (len(sequence_pair) == 2), "Each sequence pair should have a length of 2"
            # sequence of words are located in sequence_pair[0]
            word_sequence = sequence_pair[0]
            # sequence of labels are located in sequence_pair[1]
            label_sequence = sequence_pair[1]
            assert (len(word_sequence) == len(
                label_sequence)), "Length of sequence of words and sequence of labels should be the same"
            # append data to sequence_pairs
            self.sequence_pairs.append([word_sequence, label_sequence])

    def __initialize_sequencedata_attributes(self):
        """
		Initializes the SequenceData attributes from the loaded sequence_pairs list
		that is built either from file or list
		"""

        # iterate through all sentences and through all words|labels (|pos_tag)in the sequence_pairs list
        # label and pos_tag are unpack together.  then we access the label with label_sequence[0] and pos tag with
        # label_sequence[1] if len ==2
        for word_sequence, *label_sequence in self.sequence_pairs:
            for i in range(len(word_sequence)):
                # get each word in the word_sequence
                word = word_sequence[i]
                # increment num of words
                self.num_of_words += 1
                # count occurrences per word
                self.word_count[word] += 1
                # get label for the respective word
                label = label_sequence[0][i]
                # get the pos tag for the respective word
                pos_tag = None if len(label_sequence) == 1 else label_sequence[1][i]
                # check if label is None
                if label is not None:
                    # increment number of labeled words
                    self.num_labeled_words += 1
                    # count occurrences for each label
                    self.label_count[label] += 1
                    # put word in the word_label_count dict if not present
                    if not (word in self.word_label_count):
                        self.word_label_count[word] = collections.Counter()
                    # increment counter for the word|label pair
                    self.word_label_count[word][label] += 1
                if pos_tag is not None:
                    # count occurrence for each pos tag
                    self.pos_count[pos_tag] += 1
                    # put word in the word_pos_count dict if not present
                    if not (word in self.word_pos_count):
                        self.word_pos_count[word] = collections.Counter()
                    # increment counter for the word | pos pair
                    self.word_pos_count[word][pos_tag] += 1

        # sort word_label_count[word] dictionary such that the most frequent label per word appears first
        # e.g. { "set" : {"verb":10, "noun":8} }
        for word in self.word_label_count:
            self.word_label_count[word] = sorted(self.word_label_count[word].items(),
                                                 key=lambda pair: pair[1], reverse=True)
        for word in self.word_pos_count:
            self.word_pos_count[word] = sorted(self.word_pos_count[word].items(),
                                               key=lambda pair: pair[1], reverse=True)

    def save_prediction_to_file(self,pred_labels, dir_output, id="", list_tokens=False):
        # file to print all predictions
        file_name = os.path.join(dir_output, "predictions" + str(id) + ".txt")
        f1 = open(file_name, "w")
        # file to print only sentences that contain at least one wrong label after classification
        file_name = os.path.join(dir_output, "predictions_wrong" + str(id) + ".txt")
        f2 = open(file_name, "w")
        # file to print only sentences whose labels are predicted 100% correctly
        file_name = os.path.join(dir_output, "predictions_correct" + str(id) + ".txt")
        f3 = open(file_name, "w")
        # index for prediction label
        pred_idx = 0
        # list to store all true labels
        true_labels = []
        true_pos_tags = []
        # iterate through the test set
        # labels_pos: [[labels]. [pos tag]] => labels = labels_pos[0] / pos_tag = labels_pos[1]
        for (words, *labels_pos), pred_sequence in zip(self.sequence_pairs, pred_labels):
            # prediction sequence for each sentence
            for i in range(len(words)):
                # append label to the list of true labels
                true_labels.append(labels_pos[0][i])
                # append pos tag (if exist) to the list of true pos tag
                if len(labels_pos) == 2: true_pos_tags.append(labels_pos[1][i])
                # create line to print in the file
                if not list_tokens: # used in case of NN or CRF because pred_labels is a list of sentence : list[list[label]]
                    if type(words) == dict:
                        line = words[i]['t'] + " " + "POS" + " " + labels_pos[0][i] + " " + pred_sequence[i] + "\n"
                    else:
                        line = words[i] + " " + "POS" + " " + labels_pos[0][i] + " " + pred_sequence[i] + "\n"

                else: # use in case of svm: pred_labels is list of tokens
                    line = words[i] + " " + "POS" + " " + labels_pos[0][i] + " " + pred_labels[pred_idx] + "\n"
                # write to file
                f1.write(line)
                pred_idx += 1
            # separate sentences with empty lines
            f1.write("\n")
            # check if classification error occurred
            if labels_pos[0] != pred_sequence:
                for i in range(len(labels_pos[0])):
                    # create line to print to file
                    if type(words) == dict:
                        line = words[i]['t'] + " " + "POS" + " " + labels_pos[0][i] + " " + pred_sequence[i] + "\n"
                    else:
                        line = words[i] + " " + "POS" + " " + labels_pos[0][i] + " " + pred_sequence[i] + "\n"
                    f2.write(line)
                # separate sentences with empty lines
                f2.write("\n")
            else:
                for i in range(len(labels_pos[0])):
                    # create line to print to file
                    if type(words) == dict:
                        line = words[i]['t'] + " " + "POS" + " " + labels_pos[0][i] + " " + pred_sequence[i] + "\n"
                    else:
                        line = words[i] + " " + "POS" + " " + labels_pos[0][i] + " " + pred_sequence[i] + "\n"
                    f3.write(line)
                # separate sentences with empty lines
                f3.write("\n")
        # close files
        f1.close()
        f2.close()
        f3.close()
        return true_labels

    # overrides str method for SequenceData object
    def __str__(self):
        """
		String representation of sequence pairs

		@return: a string that represents the SequenceData object
		"""

        string_rep = ""
        # build a string representation such that each line is of the form:
        # word_1	label_1
        # word_2
        # (empty line)
        # ...
        # word_n	label_n
        # the second column is empty if label is None
        # empty lines separate sentences
        for sequence_num, (word_sequence, label_sequence) in enumerate(self.sequence_pairs):
            for position in range(len(word_sequence)):
                string_rep += word_sequence[position]
                if not (label_sequence[position] is None):
                    string_rep += "\t" + label_sequence[position]
                string_rep += "\n"
            if sequence_num < len(self.sequence_pairs) - 1:
                string_rep += "\n"
        return string_rep

    def get_copy(self):
        return copy.copy(self)