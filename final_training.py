import json
import os
import sys
from pprint import pprint

import numpy as np

from MinitaggerCRF_tf import MinitaggerCRF_tf
from MinitaggerSVM import MinitaggerSVM

sys.path.insert(0, 'helper')

from helper.SequenceData import SequenceData
from helper.utils import *

from MinitaggerCRF import MinitaggerCRF
from FeatureExtractor_CRF_SVM import FeatureExtractor_CRF_SVM


class Final_training(object):
    def __init__(self, parameters):
        self.validation_data_path = parameters["validation_data_path"]
        self.train_data_path = parameters["train_data_path"]
        self.test_data_path = parameters["test_data_path"]
        self.minitagger = None
        self.language = parameters["language"]
        self.model_name = parameters["model_name"]
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
        self.validation_sequence = SequenceData(self.validation_data_path,language=self.language, pos_tag=True) if self.validation_data_path is not None else None
        self.test_sequence = SequenceData(self.test_data_path,language=self.language, pos_tag=True)

        # load bitstring or embeddings data
        if self.feature_template == "embedding":
            self.feature_extractor.load_word_embeddings(self.embedding_path, self.embedding_size)

        print("Number of sentences in training dataset:", len(self.train_sequence.sequence_pairs))
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


        # equip Minitagger with the appropriate feature extractor
        self.minitagger.equip_feature_extractor(self.feature_extractor)

        self.minitagger.algorithm = "lbfgs"
        if self.best_param is not None:
            self.minitagger.c1 = self.best_param['c1']
            self.minitagger.c2 = self.best_param['c2']
            self.minitagger.epsilon = self.best_param['epsilon']
        else:
            self.minitagger.c1 = 0.1282051282051282
            self.minitagger.c2 = 0.23076923076923075
            self.minitagger.epsilon =0.0008376776400682924

        self.minitagger.all_possible_state = True
        self.minitagger.all_possible_transitions = True

        self.minitagger.extract_features(self.train_sequence, self.test_sequence, self.validation_sequence)
        if self.train_with_step:
            self.minitagger.train_with_step()
        else:
            self.minitagger.quiet = False
            self.minitagger.train()

    def crf_tf(self, best_param, train_with_step):
        self.minitagger = MinitaggerCRF_tf()
        self.minitagger.language = self.language
        self.minitagger.quiet = True
        self.minitagger.set_prediction_path(self.model_name)
        self.minitagger.set_model_path(self.model_name)
        self.minitagger.equip_feature_extractor(self.feature_extractor)

        self.minitagger.extract_features(self.train_sequence, self.test_sequence, self.validation_sequence)
        self.minitagger.train()

    def svm(self):
        self.minitagger = MinitaggerSVM()
        self.minitagger.language = self.language
        self.minitagger.quiet = True
        self.minitagger.equip_feature_extractor(self.feature_extractor)
        self.minitagger.set_prediction_path(self.model_name)
        self.minitagger.set_model_path(self.model_name)

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
            self.minitagger.quiet=False
            self.minitagger.train()
            self.minitagger.save(self.minitagger.model_path)


if __name__ == "__main__":

    #parameters = load_parameters("conll_de")
    #parameters = load_parameters("wikiner_de")
    parameters = load_parameters("new")
    training = Final_training(parameters=parameters)
    training.train()


    sys.exit()



    a = "conll"
    language = "de"
    embedding_size = 300
    if a == "new_dataset":
        filter = "0.01"
        dataset_method = "wp0"
        #train_data_path = "../../new_dataset/"+dataset_method+"/combined_"+str(filter)+".txt"
        train_data_path = "../../new_dataset/"+dataset_method+"/combined_"+dataset_method+"_"+str(filter)+".txt"
        #train_data_path = "../../new_dataset/"+dataset_method+"/combined_wp0_0.005_moreORG.txt"
        validation_data_path = "../../ner/nerc-conll2003/eng.testa"
        test_data_path = "../../ner/nerc-conll2003/eng.testb"
        feature_template = "baseline"
        #feature_template = "embedding"
        #embedding_path = "../../word_embeddings/glove"
        embedding_size = 50
        embedding_path = "../../word_embeddings/glove"
        method = "SVM"

        model_name = "_new_dataset_"+dataset_method+"_"+str(filter)+"_finale_score"

    elif a == "wikiner":
        language = "de"
        if language == "en":
            file_name = "eng"
        elif language == "de":
            file_name = "deu"
        train_data_path = "../../wikiner_dataset/aij-wikiner-"+language+"-wp2-simplified"
        #train_data_path = "../../ner/nerc-conll2003/eng-simplified.train"
        validation_data_path = "../../ner/nerc-conll2003/"+file_name+".testa"
        test_data_path = "../../ner/nerc-conll2003/"+file_name+".testb"
        feature_template = "baseline"
        embedding_path = "../../word_embeddings/glove"
        #model_name = "SVM_wikiner_emb" + str(embedding_size) + "_finale_score"
        model_name = "_wikiner_baseline_finale_score"
        method = "SVM"
    elif a == "conll":
        language = "de"
        if language == "en":
            file_name = "eng"
        elif language == "de":
            file_name = "deu"
        #train_data_path = "../../ner/nerc-conll2003/deu-simplified.train"
        train_data_path = "../../ner/nerc-conll2003/"+file_name+".train"
        #validation_data_path = "../../ner/nerc-conll2003/deu-simplified.testa"
        validation_data_path = "../../ner/nerc-conll2003/"+file_name+".testa"
        #test_data_path = "../../ner/nerc-conll2003/deu-simplified.testb"
        test_data_path = "../../ner/nerc-conll2003/"+file_name+".testb"
        feature_template = "embedding"
        feature_template = "baseline"
        embedding_path = "../../word_embeddings/glove"
        #embedding_size = 300
        method = "SVM"
        if feature_template == "baseline":
            model_name = method + "_conll_baseline_finale_score_small_feature_name"
        else:
            model_name = method + "_conll_emb" + str(embedding_size) + "_finale_score"


    selection = final_training(train_data_path, validation_data_path, test_data_path, language, model_name,
                                      feature_template, embedding_size, embedding_path)

    # elif algorithm == "CRF":
    selection.train(method)


    # if algorithm == "SVM":
    # model_name = "SVM_wikiner_emb"+str(embedding_size)+"_parameter_selection_1"
    # parameter_selection_svm(train_data_path,test_data_path, language, model_name, feature_template, embedding_size,embedding_path)
