import copy
import os
import pickle
import time
import sys

import numpy as np
import sklearn_crfsuite
from helper.model_evaluation import report_fscore_from_file

from Minitagger import Minitagger
from helper.score import Score


class MinitaggerCRF(Minitagger):
    """
	Represents the Minitagger model and can be used to train a classifier and make predictions.
	Also it includes the active learning feature
	"""
    def __init__(self):
        Minitagger.__init__(self)
        #paraneter for crf
        self.c1 = 0.07692307692307693
        self.c2 = 0.07692307692307693
        self.algorithm = "lbfgs"
        self.epsilon = 1e-5
        self.all_possible_state = False
        self.all_possible_transition = True
        self.validation_ratio = 0.05
        print("CRF (sklearn)")

    def extract_features(self, train_sequence, test_sequence, validation_sequence = None, data_to_use=20000):
        if validation_sequence is None:
            train_sequence, validation_sequence = train_sequence.split_train_validation(
                train_ratio=1 - 0.05)
            if not self.quiet:
                print("Split: ", "train: ", 1 - 0.05, "dev:", self.validation_ratio)


        # Extract feature of the training set
        if not self.quiet:
            print("Extract features")
        #split sequence pairs => easier to use
        train_tokens_sequence, self.train_labels_sequence = train_sequence.split_token_label(num_of_sentences=data_to_use)
        validation_tokens_sequence, self.validation_labels_sequence = validation_sequence.split_token_label(num_of_sentences=data_to_use)
        test_tokens_sequence, self.test_labels_sequence = test_sequence.split_token_label(num_of_sentences=data_to_use)
        self.test_sequence = test_sequence

        assert (self.feature_extractor.is_training), "In order to train, is_training flag should be True"
        self.train_features, _, _ = self.feature_extractor.extract_features_crf(train_tokens_sequence, self.train_labels_sequence, extract_all=False)

        self.__save_features(self.train_features, self.model_path, "train_features")
        if not self.quiet:
            print("Extract test features")
        self.feature_extractor.is_training = False

        if validation_sequence is not None:
            self.validation_features, _, _ = self.feature_extractor.extract_features_crf(validation_tokens_sequence,
                                                                                                      self.validation_labels_sequence,
                                                                                                      extract_all=True)
            self.__save_features(self.validation_features, self.model_path, "validation_features")

        self.test_features, _, _ = self.feature_extractor.extract_features_crf(test_tokens_sequence,
                                                                                            self.test_labels_sequence,
                                                                                            extract_all=True)
        self.__save_features(test_tokens_sequence, self.model_path, "test_features")

    def train(self):
        """
		Trains Minitagger on the given train data. If test data is given, it reports the accuracy of the trained model
		and the F1_score (macro average of f1_score of each label)
		@type train_sequence: SequenceData
		@param train_sequence: the training data set
		@type test_sequence: SequenceData
		@param test_sequence: the test data set
		"""
        # keep the training start timestamp


        if not self.quiet:
            print("Number of sentences train: ", len(self.train_labels_sequence))
            print("Number of sentences test: ", len(self.test_labels_sequence))
            print("{0} feature types".format(self.feature_extractor.num_feature_types()))
            print("\"{0}\" feature template".format(self.feature_extractor.feature_template))

        crf = sklearn_crfsuite.CRF(
            algorithm=self.algorithm,
            c1=self.c1,
            c2=self.c2,
            epsilon=self.epsilon,
            #max_iterations=10000,
            all_possible_transitions=self.all_possible_transition,
            all_possible_states=self.all_possible_state,
        #    verbose= not self.quiet
        )

        if not self.quiet:
            print("Train model")
        crf.fit(
            self.train_features,
            self.train_labels_sequence,
            X_dev = self.validation_features,
            y_dev = self.validation_labels_sequence)

        if not self.quiet:
            print("Predict")
        y_pred = crf.predict(self.test_features)
        self.test_sequence.save_prediction_to_file(y_pred, self.prediction_path, id)
        exact_score, inexact_score, conllEval = report_fscore_from_file(self.prediction_path + "/predictions"+str(id)+".txt",
                                                             wikiner=self.wikiner, quiet=True)
        if not self.quiet:
            self.display_results("Conll", conllEval)
            self.display_results("Exact", exact_score)
            self.display_results("Inexact", inexact_score)

        self.save_results(conllEval, exact_score, inexact_score,id)
        self.__save_model(crf)
        return  exact_score, inexact_score, conllEval


    def train_with_step(self):
        print("Number of sentences train: ", len(self.train_labels_sequence))
        print("Number of sentences test: ", len(self.test_labels_sequence))
        print("{0} feature types".format(self.feature_extractor.num_feature_types()))
        print("\"{0}\" feature template".format(self.feature_extractor.feature_template))

        crf = sklearn_crfsuite.CRF(
            algorithm=self.algorithm,
            c1=self.c1,
            c2=self.c2,
            epsilon=self.epsilon,
            all_possible_transitions=self.all_possible_transition,
            all_possible_states=self.all_possible_state,
          #  verbose=not self.quiet
        )

        score = Score("finale_score", None)
        param = dict()
        param["c1"] = self.c1
        param["c2"] = self.c2
        param["epsilon"] = self.epsilon

        for i in np.linspace(0,700, 50):
            start_time = time.time()
            arg = np.random.permutation(len(self.train_features))
            self.train_features = np.array(self.train_features)[arg]
            self.train_labels_sequence = np.array(self.train_labels_sequence)[arg]
            crf.max_iterations = 800
            crf.fit(
                self.train_features,
                self.train_labels_sequence,
                X_dev=self.validation_features,
                y_dev=self.validation_labels_sequence)
            y_pred = crf.predict(self.test_features)
            self.test_sequence.save_prediction_to_file(y_pred, self.prediction_path)
            exact_score, inexact_score, conllEval = report_fscore_from_file(
                self.prediction_path + "/predictions.txt",
                wikiner=self.wikiner, quiet=True)
            score.add_new_iteration(i, time.time()-start_time, conllEval,exact_score,inexact_score,param)
            print("Iteration:",i, "score: ", conllEval)

            score.save_class_to_file(self.model_path)
        self.__save_model(crf)



    def __save_features(self, data, dir, name):
        for block_num, i in enumerate(range(0,len(data), 1000)):
            with open(os.path.join(dir, name+"_" + str(block_num) + ".p"), "wb") as file:
                pickle.dump(data[i: i+500], file)

    def __load_features(self, dir, name):
        i = 0
        data =np.empty(0)
        while os.path.exists(os.path.join(dir, name +"_" + str(i) + ".p")):
            print(i)
            with open(os.path.join(dir, name + "_" + str(i) + ".p"), "rb") as file:
                data = np.append(data, pickle.load(file), axis = 0)
            i += 1
        return data

    def __save_model(self, model):
        with open(os.path.join(self.model_path,"crf_model.p"), 'wb') as fid:
            pickle.dump(model, fid)
















    def cross_validation(self, id_, data_train,feature_template, language,embedding_path, embedding_size,  data_test=None, n_fold=2):
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

        print("Parameters :")
        print("  Algorithm:", self.algorithm)
        print("         c1:", self.c1)
        print("         c2:", self.c2)
        print("    epsilon:", self.epsilon)

        parameter = dict()
        parameter["epsilon"] = self.epsilon
        parameter["c1"] = self.c1
        parameter["c2"] = self.c2
        parameter["algorithm"] = self.algorithm


        # for cross validation we want to train in quiet mode to not have x times the same information
        # but we need to restore the quiet status at the end of the cross validation
        self.quiet = True

        # data_train.sequence_pairs: [[[tokens sentence1]. [label senteces1]], [[tokens sentence2]. [label senteces12]],...]
        # 1)permute randomly all sentences
        data_train.sequence_pairs = np.random.permutation(data_train.sequence_pairs)
        # 2) do cross validation

        training_size = len(data_train.sequence_pairs)
        test_size = int(training_size / n_fold)
        score = Score("CRF_CV_" + str(id_), None)

        if data_test is not None:
            data_train.sequence_pairs = np.append(data_train.sequence_pairs, data_test.sequence_pairs, axis=0)

        for k in range(n_fold):
            print("--- Fold: {} ---".format(k+1))
            #reset features extractor:
            self.feature_extractor.reset()
            train_set, test_set = data_train.split_in_2_sequences(start=k * test_size, end=(k + 1) * test_size)
            self.extract_features(train_set, test_set)
            exact_score, inexact_score, conllEval = self.train()
            score.add_scores(conllEval, exact_score, inexact_score, parameter)
        print("Mean conll fscore: ", score.get_mean_conll_fscore())
        return score.get_mean_conll_fscore(), parameter

