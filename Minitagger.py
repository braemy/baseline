import json
import os
import sys
sys.path.insert(0, 'helper')

import numpy as np
from sklearn.metrics import confusion_matrix
from utils import create_recursive_folder

#LIBLINEAR_PATH = os.path.join(os.path.dirname(__file__), "liblinear-1.96/python")
#print(LIBLINEAR_PATH)
#sys.path.append(os.path.abspath(LIBLINEAR_PATH))

class Minitagger(object):
    """
	Represents the Minitagger model and can be used to train a classifier and make predictions.
	Also it includes the active learning feature
	"""

    def __init__(self):
        # feature extractor that is used (it is a SequenceDataFeatureExtractor object)
        self.feature_extractor = None
        # flag in order to print more/less log messages
        self.quiet = False
        # path to output directory for active learning
        self.active_output_path = ""
        # store predictions
        self.debug = False
        # path to output the predictions
        self.prediction_path = ""
        # path to output the predictions
        self.model_path = ""
        # path of the project
        self.project_dir = "."
        # language of the model
        self.language = ""
        # wikiner dataset
        self.wikiner = ""

        #Parameters:
        self.epsilon = 0.1

    def equip_feature_extractor(self, feature_extractor):
        """
		Equips the Minitagger with a feature extractor

		@type feature_extractor: SequenceDataFeatureExtractor
		@param feature_extractor: contains the feature extraction object
		"""
        self.feature_extractor = feature_extractor

    def set_is_training(self, is_training):
        self.feature_extractor.is_training=is_training

    def set_prediction_path(self, model_name):
        create_recursive_folder([self.project_dir, "predictions", self.language, model_name])
        self.prediction_path = os.path.join("predictions", self.language, model_name)

    def set_model_path(self, model_path):
        create_recursive_folder([self.project_dir, "models_path", self.language, model_path])
        self.model_path = os.path.join(self.project_dir, "models_path", self.language, model_path)

    @staticmethod
    def display_results(title, dict):
        print("==================", title, "==================")
        print("  F1-Score: ", dict["f1score"])
        print("  Precision: ", dict["precision"])
        print("  Recall: ", dict["recall"])
        print()

    @staticmethod
    def display_mean_result(title, fscore, precision, recall):
        print("==================", title,"==================")
        print("  F1-Score: ", fscore)
        print("  F1-score Mean: {:.3f}".format(np.mean(fscore)))
        print("  F1-score Standard deviation: {:.3f}".format(np.std(fscore)))
        print("  ---")
        print("  Precision: ", precision)
        print("  Precision Mean: {:.3f}".format(np.mean(precision)))
        print("  Precision Standard deviation: {:.3f}".format(np.std(precision)))
        print("  ---")
        print("  Recall: ", recall)
        print("  Recall Mean: {:.3f}".format(np.mean(recall)))
        print("  Recall Standard deviation: {:.3f}".format(np.std(recall)))
        print()

    def save_results(self, conll, exact, inexact, id=""):
        with open(os.path.join(self.model_path, "results_"+str(id)+".json"), "w") as file:
            result = dict()
            result["ConllEval"] =  conll
            result["Exact"] =  exact
            result["Inexact"] =  inexact
            json.dump(result, file, indent=2, separators=(',', ': '))

    def __debug(self, data_test, pred_labels):
        """
		Creates log files useful for debugging and prints a confusion matrix

		@type data_test: SequenceData object
		@param data_test: contains the testing data set
		@type pred_labels: list
		@param pred_labels: contains the prediction labels as they result from the classifier
		"""

        true_labels = self.__save_prediction_to_file(data_test, pred_labels)
        print()
        # find number of each label in the test set
        max_count = 0
        labels_list = list(data_test.label_count.keys())
        for label in labels_list:  # ["B", "I", "O"]:
            count = (np.array(true_labels) == label).sum()
            print("Number of " + label + " in the test set:", count)
            # find most frequent class in the test data set
            if count > max_count:
                max_count = count
                l = label
        print()
        # print accuracy of a naive baseline mode
        baseline_accuracy = "{0:.3f}".format(float(max_count) / len(true_labels) * 100)
        print("A naive model could predict always \'" + l + "\' with an accuracy of " + baseline_accuracy + "%")
        print()
        # create confusion matrix
        cm = confusion_matrix(true_labels, pred_labels, labels_list)
        print(labels_list)
        for row in cm:
            print(row)
            # plot_confusion_matrix(cm, labels_list)

