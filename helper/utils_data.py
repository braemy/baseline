import numpy as np

def  get_word(word_sequence, position):
    """
    Gets the word at the specified position

    @type word_sequence: list
    @param word_sequence: list of words in a sequence
    @type position: int
    @param position: position of word in the list
    @return: if position is valid, then return the respective word, else return _START_ or _END_
    """
    if position < 0:
        return "_START_"
    elif position >= len(word_sequence):
        return "_END_"
    else:
        return word_sequence[position]


def get_chunk(word_sequence, start, end):
    """
    return a chunk from start to end (included)
    :param word_sequence:
    :type word_sequence:
    :param start:
    :type start:
    :param end:
    :type end:
    :return:
    :rtype:
    """
    sentence = ["_START_"]
    sentence.extend(word_sequence)
    sentence.append("_END_")
    return " ".join(sentence[start + 1: end + 2])


def get_pos(pos_sequence, position):
    """
    Gets the part of speech tag at the specified position
    :param pos_sequence: list
    :type pos_sequence:  list of pos in a sequence
    :param position: int
    :type position: position of the tag in the list
    :return: if position is valid, then return the respective word, esle return _START_ or _END_
    """
    if position < 0:
        return "_START_"
    elif position >= len(pos_sequence):
        return "_END_"
    else:
        return pos_sequence[position]


def is_capitalized(word):
    """
    Checks if the given word is capitalized

    @type word: str
    @param word: word to be checked
    """
    return word[0].isupper()


def get_prefix(word, length):
    """
    Gets a prefix of the word up to the given length
    If the length is greater than the word length, the prefix is padded

    @type word: str
    @param word: word where the prefix is built from
    @type length: int
    @param length: length of prefix
    @return: prefix
    """
    if length <= 0:
        return ""
    if length <= len(word):
        return word[:length]
    else:
        return word.ljust(length, "*")


def get_suffix(word, length):
    """
    Gets a suffix of the word up to the given length
    If the length is greater than the word length, the suffix is padded

    @type word: str
    @param word: word where the suffix is built from
    @type length: int
    @param length: length of suffix
    @return: suffix
    """
    if length <= 0:
        return ""
    if length <= len(word):
        start = len(word) - length
        return word[start:]
    else:
        return word.rjust(length, "*")


def is_all_nonalphanumeric(word):
    """
    Checks if the all chars in a word are nonalphanumeric
    If at least one char is alphanumeric, this should return False

    @type word: str
    @param word: word to be checked
    @return: True of False
    """
    for char in word:
        if char.isalnum():
            return False
    return True


def has_numbers(inputString):
    return any(char.isdigit() for char in inputString)


def is_float(word):
    """
    Checks if the word be converted to a float

    @type word: str
    @param word: word to be checked
    @return: True of False
    """
    try:
        float(word)
        return True
    except ValueError:
        return False

def is_int(word):
    """
    Checks if the word be converted to a float

    @type word: str
    @param word: word to be checked
    @return: True of False
    """
    try:
        int(word)
        return True
    except ValueError:
        return False

def is_prediction_file(data_path):
    """
    Checks whether the given file is a prediction file (i.e. contains prediction labels) or not
    @type data_path: str
    @param data_path: path to file
    @return: True of False for is_prediction and is_not_prediction variables
    """
    # initialize variables
    is_prediction = False
    is_not_prediction = False
    # open file
    with open(data_path, "r") as input_file:
        for line in input_file:
            # split line into tokens
            tokens = line.split()
            if tokens:
                # each line should not contain more than 3 items
                assert (len(tokens) < 4), "The file should contain no more than 3 columns"
                # in case the row has less than 3 columns, then the file is not a prediction file
                if len(tokens) < 3:
                    is_not_prediction = True
                # in the the row has 3 columns, then the file is a prediction file
                else:
                    is_prediction = True
    return is_prediction, is_not_prediction


def recover_original_data(data_path, sequence_pairs):
    """
    Recovers the original data from the given prediction file. It measures the prediction accuracy and extracts the
    original word sequences and label sequences

    @type data_path: str
    @param data_path: path to prediction file
    @type sequence_pairs: list
    @param sequence_pairs: list that contains all pairs of sentences (also referred as sequences) and labels
    @return: a list of sequence of words with their respective labels, per word accuracy and per sequence accuracy
    """
    # initialize variables
    num_labels = 0
    num_sequences = 0
    num_correct_labels = 0
    num_correct_sequences = 0
    with open(data_path, "r") as input_file:
        # sequence of workds in each sentence
        word_sequence = []
        # gold/original labels for each word in each sentence
        gold_label_sequence = []
        # prediction labels for each word in each sentence
        pred_label_sequence = []
        for line in input_file:
            # split line into tokens
            tokens = line.split()
            # check if line is not empty
            if tokens:
                # a label exists
                num_labels += 1
                # the word is the first token
                word = tokens[0]
                # the original label is the second token
                gold_label = tokens[1]
                # the prediction label is the third token
                pred_label = tokens[2]
                # check if prediction equals to real label
                if pred_label == gold_label:
                    num_correct_labels += 1
                # build the sequence of words, labels, and predictions for each sentence
                word_sequence.append(word)
                gold_label_sequence.append(gold_label)
                pred_label_sequence.append(pred_label)
            # line is empty
            else:
                # count number of sequences (=sentences)
                num_sequences += 1
                # check if word_sequence is empty
                if word_sequence:
                    sequence_pairs.append([word_sequence, gold_label_sequence])
                # check if we predicted correctly the whole sequence
                if pred_label_sequence == gold_label_sequence:
                    num_correct_sequences += 1
                # flush lists for next sequence
                word_sequence = []
                gold_label_sequence = []
                pred_label_sequence = []
        # here is the case where the file does not end with an empty line
        # repeat the process for the last sequence of the file
        if word_sequence:
            num_sequences += 1
            sequence_pairs.append([word_sequence, gold_label_sequence])
            if pred_label_sequence == gold_label_sequence:
                num_correct_sequences += 1
    # calculate per instance (=word) accuracy and per sequence (=sentence) accuracy
    per_instance_accuracy = float(num_correct_labels) / num_labels * 100
    per_sequence_accuracy = float(num_correct_sequences) / num_sequences * 100
    return per_instance_accuracy, per_sequence_accuracy


def analyze_data(data_path):
    """
    Analyze the given data file and report some information about the file content

    @type data_path: str
    @param data_path: path to file used for analysis
    """
    # Check whether this data is a prediction file or not
    is_prediction, is_not_prediction = is_prediction_file(data_path)
    assert (is_prediction or is_not_prediction) and not (is_prediction and is_not_prediction), \
        "The file should be either a prediction file or not, i.e. it should contain either 2 or 3 columns"
    sequence_pairs = []
    # if prediction, recover the original data and also compute accuracy
    per_instance_accuracy = -1
    per_sequence_accuracy = -1
    if is_prediction:
        per_instance_accuracy, per_sequence_accuracy = recover_original_data(data_path, sequence_pairs)
    # Construct sequence data
    data = SequenceData(sequence_pairs) if is_prediction else SequenceData(data_path)
    if is_prediction:
        print("A prediction data file:", data_path)
    else:
        print("A non-prediction data file:", data_path)
    print("{0} sequences (average length: {1:.1f})".format(
        len(data.sequence_pairs), data.get_sequence_average_length()))
    print("{0} words".format(data.num_of_words))
    print("{0} labeled words".format(data.num_labeled_words))
    print("{0} word types".format(len(data.word_count)))
    print("{0} label types".format(len(data.label_count)))
    if is_prediction:
        print("Per-instance accuracy: {0:.3f}%".format(per_instance_accuracy))
        print("Per-sequence accuracy: {0:.3f}%".format(per_sequence_accuracy))


def pad_batch(features,features_embedding_indexes, labels, lengths, sub_sequence_index, o_label_num):
    sentences_features = np.array(features)[sub_sequence_index]
    sentences_features_embedding_indexes = np.array(features_embedding_indexes)[sub_sequence_index]
    sentences_features_list = [0] * len(sentences_features)
    sentences_features_embedding_indexes_list = [0] * len(sentences_features)
    sequences_length = np.array(lengths)[sub_sequence_index]

    max_length = max(sequences_length)
    num_features = len(features[0][0])
    padding_value = [0]* num_features
    num_embedding_indexes = len(features_embedding_indexes[0][0])
    padding_embedding_indexes_value = [0] * num_embedding_indexes
    for sentence_id in range(0, len(sentences_features)):
        sentence_f = sentences_features[sentence_id]
        sentence_embedding_indexes = sentences_features_embedding_indexes[sentence_id]
        #sent_array = [0] * len(sentence_f)
        #for word_id in range(0, len(sentence_f)):
            #word_f = sentence_f[word_id]
            #tmp = [0] * num_features

            #for k,v in enumerate(word_f): #TODO remove
            #    tmp[k] = v
            #for k, v in word_f.items():
            #    tmp[k] = v
            #sent_array[word_id] = tmp
        #sent_array = pad_list(sent_array, max_length, padding_value)
        sent_array = pad_list(sentence_f, max_length, padding_value)
        sentence_embedding_indexes = pad_list(sentence_embedding_indexes, max_length, padding_embedding_indexes_value)

        sentences_features_list[sentence_id] = sent_array
        sentences_features_embedding_indexes_list[sentence_id] = sentence_embedding_indexes

    labels = np.array(labels)[sub_sequence_index]
    labels = [pad_list(label_sent, max_length, o_label_num) for label_sent in labels]
    return sentences_features_list,sentences_features_embedding_indexes_list, labels, sequences_length


def pad_list(old_list, padding_size, padding_value):
    '''
    http://stackoverflow.com/questions/3438756/some-built-in-to-pad-a-list-in-python
    Example: pad_list([6,2,3], 5, 0) returns [6,2,3,0,0]
    '''
    assert padding_size >= len(old_list), "padding_size = " + str(padding_size) + "  len(old_list) = " + str(len(old_list))
    if padding_size-len(old_list) == 0:
        return old_list
    return old_list + [padding_value] * (padding_size-len(old_list))