import datetime
import math
import os
import pickle
import time
import warnings

import numpy as np

from Minitagger import Minitagger
from helper.model_evaluation import report_fscore_from_file
from helper.score import Score
from liblinear.python import liblinearutil


class MinitaggerSVM(Minitagger):
    """
    Represents the Minitagger model and can be used to train a classifier and make predictions.
    Also it includes the active learning feature
    """

    def __init__(self):

        Minitagger.__init__(self)
        # it stores a trained liblinearutil
        self.__liblinear_model = None
        # Parameters:
        self.epsilon = 0.1
        self.cost = 0.1

        print("SVM (liblinear)")

    def extract_features(self, data_train, data_test, validation_sequence=None, num_data=None):
        if data_train is None:  # data_train is none ==> we are going to predict
            # make sure is_training is false
            assert (not self.feature_extractor.is_training), "In order to train, is_training flag should be True"
        else:
            # Extract features only for labeled instances from data_train
            self.data_train = data_train.get_copy()
            self.features_list_train, self.label_list_train, _ = self.feature_extractor.extract_features_svm(
                self.data_train, extract_all=False)

        self.data_test = data_test
        if num_data is not None:
            arg = np.random.permutation(len(data_train.sequence_pairs))[:int(num_data)]
            self.data_train.sequence_pairs = np.array(data_train.sequence_pairs)[arg]

        # self.feature_extractor.is_training = False
        self.features_list_test, self.label_list_test, _ = self.feature_extractor.extract_features_svm(self.data_test,
                                                                                                       extract_all=True)
        # if validation_sequence is not None:
        #    self.data_validation = validation_sequence
        #    self.features_list_validation, self.label_list_validation, _ = self.feature_extractor.extract_features_svm(
        #        self.data_validation,
        #        extract_all=True)

    def train(self):
        """
        Trains Minitagger on the given train data. If test data is given, it reports the accuracy of the trained model
        and the F1_score (macro average of f1_score of each label)
        @type data_train: SequenceData
        @param data_train: the training data set
        @type data_test: SequenceData
        @param data_test: the test data set
        """

        # keep the training start timestamp
        start_time = time.time()

        # print some useful information about the data
        if not self.quiet:
            print("{0} labeled words (out of {1})".format(len(self.label_list_train), self.data_train.num_of_words))
            print("{0} label types".format(len(self.data_train.label_count)))
            print("label types: ", self.data_train.label_count)
            print("{0} word types".format(len(self.data_train.word_count)))
            print("\"{0}\" feature template".format(self.feature_extractor.feature_template))
            print("{0} feature types".format(self.feature_extractor.num_feature_types()))
        # define problem to be trained using the parameters received from the feature_extractor
        problem = liblinearutil.problem(self.label_list_train, self.features_list_train)
        # train the model (-q stands for quiet = True in the liblinearutil)
        #        self.__liblinear_model = liblinearutil.train(problem, liblinearutil.parameter(" -q -p " + str(self.epsilon) + " -c " +str(self.cost)))
        self.__liblinear_model = liblinearutil.train(problem, liblinearutil.parameter(" -q"))

        # training is done, set is_training to False, so that prediction can be done
        self.feature_extractor.is_training = False

        # print some useful information
        if not self.quiet:
            num_seconds = int(math.ceil(time.time() - start_time))
            # how much did the training last
            print("Training time: {0}".format(str(datetime.timedelta(seconds=num_seconds))))
            # perform prediction on the data_test and report accuracy
        if self.data_test is not None:
            quiet_value = self.quiet
            self.quiet = True
            pred_labels, acc = self.predict()
            self.quiet = quiet_value

            self.data_test.save_prediction_to_file(pred_labels, self.prediction_path)
            exact_score, inexact_score, conllEval = report_fscore_from_file(self.prediction_path + "/predictions.txt",
                                                                            wikiner=self.wikiner)
            # create some files useful for debugging
            if self.debug:
                self.__debug(self.data_test, pred_labels)
        if not self.quiet:
            self.display_results("Conll", conllEval)
            self.display_results("Exact", exact_score)
            self.display_results("Inexact", inexact_score)

        self.save_results(conllEval, exact_score, inexact_score)

        return exact_score, inexact_score, conllEval

    def cross_validation(self, id_, data_train, feature_template, language, embedding_path, embedding_size,
                         data_test=None, n_fold=5):
        """
        compute the cross validation on the data_train.
        It report the f1_score of each fold and the average with the standard deviation
        If data_test is given, it reports the accuracy on the test set and
        the F1 score (macro average f1_score of each label)

        :type data_train: SequenceData
        :param data_train: the training data_set
        :type data_test: SequenceData
        :param data_test: the test data_set
        :param n_fold: int
        :type n_fold: number of fold for the CV
        """
        start_time = time.time()
        print("======== Cross Validation " + str(id_) + " =========")
        print("Parameter:")
        print("    epsilon:", self.epsilon)
        print("    cost:", self.cost)
        parameter = dict()
        parameter["epsilon"] = self.epsilon
        parameter["cost"] = self.cost

        training_size = len(data_train.sequence_pairs)
        self.quiet = True
        assert (n_fold >= 2), "n_fold must be at least 2"
        if n_fold > training_size:
            n_fold = training_size
            warnings.warn("n_fold can not be bigger than the size of the training set. n-fold is replace by the size "
                          "of the training set")

        # data_train.sequence_pairs: [[[tokens sentence1]. [label senteces1]], [[tokens sentence2]. [label senteces12]],...]
        # 1)permute randomly all sentences
        data_train.sequence_pairs = np.random.permutation(data_train.sequence_pairs)
        # 2) do cross validation

        test_size = int(training_size / n_fold)
        score = Score("SVM_CV_" + str(id_), None)

        for k in range(n_fold):
            print("--- Fold: {} ---".format(k + 1))
            # reset features extractor:
            self.feature_extractor.reset()
            train_set, test_set = data_train.split_in_2_sequences(start=k * test_size, end=(k + 1) * test_size)
            self.extract_features(train_set, test_set)
            exact_score, inexact_score, conllEval = self.train()
            score.add_scores(conllEval, exact_score, inexact_score, parameter)
        print("Mean conll fscore: ", score.get_mean_conll_fscore())
        return score.get_mean_conll_fscore(), parameter

    def save(self, model_path):
        """
        Saves the model as a directory at the given path
        @type model_path: str
        @param model_path: path to save the trained model
        """
        # if-else statement added on 06.02.2017
        pickle.dump(self.feature_extractor, open(os.path.join(self.model_path, "feature_extractor"), "wb"),
                    protocol=pickle.HIGHEST_PROTOCOL)
        # save trained model in the model_path directory
        liblinearutil.save_model(os.path.join(self.model_path, "liblinear_model"), self.__liblinear_model)

    def load(self, model_path):
        """
        Loads the model from the directory at the given path
        @type model_path: str
        @param model_path: path to load the trained model
        """

        # load feature_extractor object (used to extract features for the test set)
        # if-else statement added on 06.02.2017

        # if (self.__feature_extractor.feature_template == "relational") and (self.__feature_extractor.parser_type == "spacy"):
        # 	print("Relational model with spaCy parser cannot be loaded")
        # else:
        # 	self.__feature_extractor = pickle.load(open(os.path.join(model_path, "feature_extractor"), "rb"))
        # 	# load trained model
        # 	self.__liblinear_model = liblinearutil.load_model(os.path.join(model_path, "liblinear_model"))
        try:
            print(self.model_path)
            self.feature_extractor = pickle.load(open(os.path.join(self.model_path, "feature_extractor"), "rb"))
            self.feature_extractor.save_json_format(os.path.join(self.model_path, "feature_extractor_json"))
            # load trained model
            self.__liblinear_model = liblinearutil.load_model(os.path.join(self.model_path, "liblinear_model"))
        except:
            raise Exception("No files found in the model path " + self.model_path)

    def predict(self):
        """
        Predicts tags in the given data
        It reports the accuracy if the data is fully labeled
        @type data_test: SequenceData
        @param data_test: the test data set
        @return: the predicted labels, the accuracy, the f1_score
        """

        # keep starting timestamp
        start_time = time.time()
        assert (not self.feature_extractor.is_training), "In order to predict, is_training should be False"

        # Extract features on all instances (labeled or unlabeled) of the test set

        # pass them to liblinearutil for prediction
        pred_labels, (acc, _, _), _ = liblinearutil.predict(self.label_list_test, self.features_list_test,
                                                            self.__liblinear_model, "-q")

        # print some useful information
        if not self.quiet:
            num_seconds = int(math.ceil(time.time() - start_time))
            # estimate prediction time
            print("Prediction time: {0}".format(str(datetime.timedelta(seconds=num_seconds))))

        # convert predicted labels from integer IDs to strings.
        pred_labels = self.convert_prediction(pred_labels, self.data_test)
        # for i, label in enumerate(pred_labels):
        #    pred_labels[i] = self.feature_extractor.get_label_string(label)
        self.__save_prediction_to_file(self.data_test, pred_labels)

        return pred_labels, acc

    def convert_prediction(self, predictions, true_label):

        sequence_prediction = [""] * len(true_label.sequence_pairs)
        pred_pos = 0
        for i, (sentence, _, _) in enumerate(true_label.sequence_pairs):
            prediction_sentence = [""] * len(sentence)
            for pos, _ in enumerate(sentence):
                prediction_sentence[pos] = self.feature_extractor.get_label_string(int(predictions[pred_pos]))
                pred_pos += 1

            sequence_prediction[i] = prediction_sentence
        return sequence_prediction

    def train_with_step(self, data_train, data_test):
        print("number of feature:", len(self.feature_extractor._map_feature_str2num.keys()))

        score = Score("finale_score", None)
        param = dict()
        param["epsilon"] = self.epsilon

        for i in np.linspace(10, len(data_train.sequence_pairs), 50):
            self.feature_extractor.reset()
            start_time = time.time()
            self.extract_features(data_train, data_test, num_data=i)

            problem = liblinearutil.problem(self.label_list_train, self.features_list_train)

            self.__liblinear_model = liblinearutil.train(problem, liblinearutil.parameter("-q -p " + str(self.epsilon)))
            pred_labels, acc = self.predict()
            self.data_test.save_prediction_to_file(pred_labels, self.prediction_path)
            exact_score, inexact_score, conllEval = report_fscore_from_file(self.prediction_path + "/predictions.txt",
                                                                            wikiner=self.wikiner, quiet=True)
            score.add_new_iteration(i, time.time() - start_time, conllEval, exact_score, inexact_score, param)
            print("Iteration:", time.time() - start_time, "score: ", conllEval)

            score.save_class_to_file(self.model_path)
            # self.save(self.model_path)

    def __save_prediction_to_file(self, data_test, pred_labels):
        # file to print all predictions
        file_name = os.path.join(self.prediction_path, "predictions.txt")
        f1 = open(file_name, "w", encoding='utf-8')
        # file to print only sentences that contain at least one wrong label after classification
        file_name = os.path.join(self.prediction_path, "predictions_wrong.txt")
        f2 = open(file_name, "w", encoding='utf-8')
        # file to print only sentences whose labels are predicted 100% correctly
        file_name = os.path.join(self.prediction_path, "predictions_correct.txt")
        f3 = open(file_name, "w", encoding='utf-8')
        # index for prediction label
        pred_idx = 0
        # list to store all true labels
        true_labels = []
        true_pos_tags = []
        # iterate through the test set
        # labels_pos: [[labels]. [pos tag]] => labels = labels_pos[0] / pos_tag = labels_pos[1]
        for words, *labels_pos in data_test.sequence_pairs:
            # prediction sequence for each sentence
            pred_sequence = []
            for i in range(len(words)):
                # append label to the prediction sequence
                pred_sequence = pred_labels[pred_idx]
                # append label to the list of true labels
                true_labels.append(labels_pos[0][i])
                # append pos tag (if exist) to the list of true pos tag
                if len(labels_pos) == 2: true_pos_tags.append(labels_pos[1][i])
                # create line to print in the file
                line = words[i] + " " + labels_pos[0][i] + " " + pred_sequence[i] + "\n"
                # write to file
                f1.write(line)

            pred_idx += 1
            # separate sentences with empty lines
            f1.write("\n")
            # check if classification error occurred
            if labels_pos[0] != pred_sequence:
                for i in range(len(labels_pos[0])):
                    # create line to print to file
                    line = words[i] + " " + labels_pos[0][i] + " " + pred_sequence[i] + "\n"
                    f2.write(line)
                # separate sentences with empty lines
                f2.write("\n")
            else:
                for i in range(len(labels_pos[0])):
                    # create line to print to file
                    line = words[i] + " " + labels_pos[0][i] + " " + pred_sequence[i] + "\n"
                    f3.write(line)
                # separate sentences with empty lines
                f3.write("\n")
        # close files
        f1.close()
        f2.close()
        f3.close()
        return true_labels
