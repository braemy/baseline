import argparse
import json
import sys
from pprint import pprint

from MinitaggerSVM import MinitaggerSVM
from helper import utils

sys.path.insert(0, 'helper')

from helper.SequenceData import SequenceData
from helper.utils import *

from MinitaggerCRF import MinitaggerCRF
from FeatureExtractor_CRF_SVM import FeatureExtractor_CRF_SVM

class Final_training(object):
    def __init__(self, parameters):
        self.validation_data_path = parameters["validation_data_path"] if "validation_data_path" in parameters else None
        self.train_data_path = parameters["train_data_path"]
        self.test_data_path = parameters["test_data_path"] if "test_data_path" in parameters else None
        self.minitagger = None
        self.language = parameters["language"]
        self.model_name = parameters["model_name"] + "_" + utils.get_current_time_in_miliseconds()
        self.feature_template = parameters["feature_template"]
        self.method = parameters["method"]
        self.best_param = parameters["best_param"] if "best_param" in parameters else None
        self.train_with_step = parameters["train_with_step"] if "train_with_step" in parameters else None
        self.parameters = parameters

        create_recursive_folder(["parameters", self.model_name])

        if self.feature_template == "embedding":
            self.embedding_size = parameters["embedding_size"]
            self.embedding_path = parameters["embedding_path"]

        pprint(parameters, depth=2)

        #self.CV = CV

    def train(self):
        # initialize feature extractor with the right feature template
        self.feature_extractor = FeatureExtractor_CRF_SVM(self.feature_template, self.language,
                                                         self.embedding_size if self.feature_template=="embedding" else None)

        self.train_sequence = SequenceData(self.train_data_path,language=self.language, pos_tag=True)
        self.validation_sequence = SequenceData(self.validation_data_path,language=self.language, pos_tag=True) if self.validation_data_path else None
        self.test_sequence = SequenceData(self.test_data_path,language=self.language, pos_tag=True) if self.test_data_path else None

        # load bitstring or embeddings data
        if self.feature_template == "embedding":
            if self.test_sequence and self.validation_sequence:
                vocabulary = self.train_sequence.vocabulary.union(self.test_sequence.vocabulary).union(self.validation_sequence.vocabulary)
            else:
                vocabulary = self.train_sequence.vocabulary
            self.feature_extractor.load_word_embeddings(self.embedding_path, self.embedding_size, vocabulary)

        print("Number of sentences in training dataset:", len(self.train_sequence.sequence_pairs))
        if self.test_sequence:
            print("Number of sentences in testing dataset:", len(self.test_sequence.sequence_pairs))

        if self.method == "CRF":
            self.crf()
        elif self.method == "SVM":
            self.svm()
        elif self.method == "CRF_TF":
            self.crf_tf()
        else:
            raise("Method unknown")

    def crf(self):
        self.minitagger = MinitaggerCRF()
        self.minitagger.language = self.language
        self.minitagger.quiet = True

        self.minitagger.set_prediction_path(self.model_name)
        self.minitagger.set_model_path(self.model_name)

        parameter_file = os.path.join(self.minitagger.model_path, "parameters.yml")
        with open(parameter_file, "w", encoding="utf-8") as file:
            yaml.dump(self.parameters, file)
        print("model name:", self.minitagger.model_path)

        # equip Minitagger with the appropriate feature extractor
        self.minitagger.equip_feature_extractor(self.feature_extractor)

        self.minitagger.algorithm = "lbfgs"
        if self.best_param is not None:
            self.minitagger.c1 = self.best_param['c1']
            self.minitagger.c2 = self.best_param['c2']
            self.minitagger.epsilon = self.best_param['epsilon_crf']
        else:
            self.minitagger.c1 = 0.1282051282051282
            self.minitagger.c2 = 0.23076923076923075
            self.minitagger.epsilon =0.0008376776400682924

        self.minitagger.all_possible_state = True
        self.minitagger.all_possible_transitions = True

        if self.train_with_step:
            self.minitagger.extract_features(self.train_sequence, self.test_sequence, self.validation_sequence)
            self.minitagger.train_with_step()
        else:
            if self.test_sequence:
                self.minitagger.extract_features(self.train_sequence, self.test_sequence, self.validation_sequence)
                self.minitagger.quiet = False
                self.minitagger.train(max_iteration=60)
            else:
                self.minitagger.cross_validation(0, self.train_sequence, n_fold=3)

    def crf_tf(self):
        self.minitagger = MinitaggerCRF_tf()
        self.minitagger.language = self.language
        self.minitagger.quiet = True
        self.minitagger.set_prediction_path(self.model_name)
        self.minitagger.set_model_path(self.model_name)
        self.minitagger.equip_feature_extractor(self.feature_extractor)

        #self.minitagger.extract_features(self.train_sequence, self.test_sequence, self.validation_sequence)
        self.minitagger.train()

    def svm(self):
        self.minitagger = MinitaggerSVM()
        self.minitagger.language = self.language
        self.minitagger.quiet = True
        self.minitagger.equip_feature_extractor(self.feature_extractor)
        self.minitagger.set_prediction_path(self.model_name)
        self.minitagger.set_model_path(self.model_name)

        parameter_file = os.path.join(self.minitagger.model_path, "parameters.yml")
        with open(parameter_file, "w", encoding="utf-8") as file:
            yaml.dump(self.parameters, file)

        if self.best_param is not None:
            self.minitagger.epsilon = self.best_param['epsilon']
            self.minitagger.cost = self.best_param['cost']
        else:
            self.minitagger.epsilon =7.196856730011514e-06
            self.minitagger.cost =  0.016378937069540647

        if self.train_with_step:
            self.minitagger.train_with_step(self.train_sequence, self.test_sequence)
        else:
            self.minitagger.extract_features(self.train_sequence, self.test_sequence)
            self.minitagger.quiet = False
            self.minitagger.train()
            self.minitagger.save(self.minitagger.model_path)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--conf", type=str, help="name of the configuration in the parameter file",
                           required=True)

    #parameters = load_parameters("conll_de")
    #parameters = load_parameters("wikiner_de")
    parsed_args = argparser.parse_args()
    parameters = load_parameters(parsed_args.conf)
    training = Final_training(parameters=parameters)
    training.train()


