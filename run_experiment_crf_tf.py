import argparse
import json
import sys
import helper.constants as constants
from pprint import pprint

from MinitaggerSVM import MinitaggerSVM
from helper import utils

sys.path.insert(0, 'helper')

from helper.SequenceData import SequenceData
from helper.utils import *


from MinitaggerCRF import MinitaggerCRF
from MinitaggerCRF_tf import MinitaggerCRF_tf
from FeatureExtractor_CRF_SVM import FeatureExtractor_CRF_SVM
from FeatureExtractor_CRF_tf import FeatureExtractor_CRF_tf
from helper.constants import dataset_type
from itertools import product


class Run_experiment_crf_tf(object):

    def run_experiment(self, languages):
        embeddings_type = ['fasttext', 'fasttext_noOOV']
        embedding_language = ['target', 'source']
        combination = product(languages, embeddings_type, embedding_language)
        create_folder_if_not_exists(os.path.join("log"))
        self.experiment_timestamp = utils.get_current_time_in_miliseconds()
        log_file = os.path.join("log", "experiment-{}.log".format(self.experiment_timestamp))

        for language, emb_type, emb_language in combination:
            print(language, emb_type, emb_language)
            self.set_parameters(language, emb_type, emb_language)
            sys.stdout = open(os.path.join("log", self.model_name), "w")
            pprint(self.parameters)
            with open(log_file, "a") as file:
                file.write("Experiment: {}\n".format(self.model_name))
                file.write("Start time:{}\n".format(utils.get_current_time_in_miliseconds()))
                file.write("-------------------------------------\n\n")
            self.train()

    def set_parameters(self, language, emb_type, emb_language):
        parameters = dict()
        if language not in ['en', 'fr', 'it', 'de']:
            file_name = "../../new_dataset/{}/wp3/combined_wp3_1.0".format(language)
            parameters['train_data_path'] = "{}.train".format(file_name)
            parameters['validation_data_path'] = "{}.valid".format(file_name)
            parameters['test_data_path'] = "{}.test".format(file_name)
        if emb_language == 'source':
            parameters['language'] = constants.MAPPING_LANGUAGE[language]

        else:
            parameters['language'] = language

        parameters['model_name'] = '{0}_{1}_{2}_{3}'.format(language, emb_type, emb_language,
                                                                self.experiment_timestamp)
        parameters['embedding_type'] = emb_type
        if emb_type == 'polyglot':
            parameters['embedding_size'] = 64
        elif 'fasttext' in emb_type:
            parameters['embedding_size'] = 300
        else:
            raise("Uknown embedding type")

        parameters['embedding_path'] = "../../word_embeddings"
        parameters['learning_rate'] = 0.0004
        self.validation_data_path = parameters["validation_data_path"]
        self.train_data_path = parameters["train_data_path"]
        self.test_data_path = parameters["test_data_path"]
        self.minitagger = None
        self.language = parameters["language"]
        self.model_name = parameters["model_name"]
        self.parameters = parameters
        self.embedding_size = parameters['embedding_size']


    def train(self):
        self.train_sequence = SequenceData(self.train_data_path,language=self.language, pos_tag=True)
        self.validation_sequence = SequenceData(self.validation_data_path,language=self.language, pos_tag=True) if self.validation_data_path else None
        self.test_sequence = SequenceData(self.test_data_path,language=self.language, pos_tag=True) if self.test_data_path else None

        self.feature_extractor = FeatureExtractor_CRF_tf("embedding", self.language,
                                                             self.embedding_size, self.train_sequence.part_of_speach_set )

        # load bitstring or embeddings data
        if self.test_sequence and self.validation_sequence:
            vocabulary = self.train_sequence.vocabulary.union(self.test_sequence.vocabulary).union(
                self.validation_sequence.vocabulary)
        else:
            vocabulary = self.train_sequence.vocabulary.union(self.test_sequence.vocabulary)
        self.feature_extractor.load_word_embeddings(self.parameters, vocabulary)

        print("Number of sentences in training dataset:", len(self.train_sequence.sequence_pairs))
        print("Number of sentences in testing dataset:", len(self.test_sequence.sequence_pairs))

        self.crf_tf()


    def crf_tf(self):
        self.minitagger = MinitaggerCRF_tf(self.parameters,self.train_sequence, self.test_sequence, self.validation_sequence)

        #self.feature_extractor.morphological_features = "embeddings"

        self.feature_extractor.token_features2 = True
        self.feature_extractor.morphological_features = "embedding"
        self.feature_extractor.token_features1 = True
        self.feature_extractor.token_features0 = True
        self.minitagger.equip_feature_extractor(self.feature_extractor)

        self.minitagger.extract_features()
        self.minitagger.train()



if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    #argparser.add_argument("--language",nargs='+', type=str, help="list of names of the configuration in the parameter file",
    #                       required=True)
    #parsed_args = argparser.parse_args()




    training = Run_experiment_crf_tf()
    training.run_experiment(constants.it_dialect)


