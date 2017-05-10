import os
import numpy as np
import json
import time
import sys

from MinitaggerSVM import MinitaggerSVM

from score import Score

from final_training import final_training

sys.path.insert(0, 'helper')

from SequenceData import SequenceData
from utils import *

from MinitaggerCRF import MinitaggerCRF
from FeatureExtractor_CRF_SVM import FeatureExtractor_CRF_SVM

class Parameter_selection(object):
    def __init__(self):
        self.minitagger=None

    def parameter_selection_svm(self, train_data_path, test_data_path, language, model_name, feature_template, embedding_size=None, embedding_path=None, number_of_trial=20, seed=123456):
        np.random.seed(seed=seed)

        minitagger = MinitaggerSVM()

        sequence_data = SequenceData(train_data_path, pos_tag=True)
        if test_data_path is not None:
            test_sequence = SequenceData(test_data_path, pos_tag=True)
        minitagger.language = language
        minitagger.set_prediction_path(model_name)
        minitagger.set_model_path(model_name)
        #minitagger.quiet = True


        # list of parameter:
        epsilon = np.logspace(-6, 0, number_of_trial * 3)
        cost = np.logspace(-5, 4, number_of_trial * 3)


        self.display_info(len(sequence_data.sequence_pairs), language, model_name, feature_template, embedding_size, embedding_path)
        infos = dict()
        infos["algorithm"] = "SVM"
        infos["train_data_path"] = train_data_path
        infos["test_data_path"] = test_data_path
        infos["language"] = language
        infos["mpdel_name"] = model_name
        infos["feature_template"] =  feature_template
        infos["embedding_size"] = embedding_size
        infos["embedding_path"] = embedding_path

        parameter_aready_tested = []

        # initialize feature extractor with the right feature template
        feature_extractor = FeatureExtractor_CRF_SVM(feature_template, language,
                                                         embedding_size if embedding_size else None)
        # load bitstring or embeddings data
        if feature_template == "embedding":
            feature_extractor.load_word_embeddings(embedding_path, embedding_size)
        # equip Minitagger with the appropriate feature extractor
        minitagger.equip_feature_extractor(feature_extractor)
        minitagger.extract_features(sequence_data,test_sequence)

        score = Score("SVM_parameter_selection", infos)

        for i in range(number_of_trial):

            ok=False
            while not ok:
                parameter = dict()
                minitagger.epsilon = np.random.choice(epsilon)
                minitagger.cost = np.random.choice(cost)
                parameter["epsilon"] = minitagger.epsilon
                if parameter not in parameter_aready_tested:
                    parameter_aready_tested.append(parameter)
                    ok = True

            mean_fscore_conll, param = minitagger.cross_validation(i+1, sequence_data, feature_template, language,
                                                                   embedding_path, embedding_size, data_test=None, n_fold=5)
            score.add_scores(mean_fscore_conll, None, None, param)
            score.save_result_to_file(infos, minitagger.model_path)

        score.display_results()

        #Retrained with best parameters:
        print("============================")
        print("Trained with best parameters")
        print("============================")
        _,_,_,best_param,_  = score.get_max_conll_fscore()

        validation_data_path = "../../ner/nerc-conll2003/eng-simplified.testa"
        test_data_path = "../../ner/nerc-conll2003/eng-simplified.testb"
        model_name = model_name+"_finale_score"

        selection = final_training(train_data_path, validation_data_path, test_data_path, language, model_name,
                                   feature_template, embedding_size, embedding_path)
        selection.final_training("svm", best_param )


    def parameter_selection_crf(self, train_data_path, validation_data_path, test_data_path, language, model_name, feature_template, embedding_size=None, embedding_path=None, number_of_trial=10, seed=123456):
        np.random.seed(seed=seed)
        data_to_use = 20000
        self.minitagger = MinitaggerCRF()

        self.sequence_data = SequenceData(train_data_path, pos_tag=True)
        if validation_data_path is not None:
            self.validation_sequence = SequenceData(validation_data_path, pos_tag=True)
        else:
            self.validation_sequence = None
        self.test_sequence = SequenceData(test_data_path, pos_tag=True)
        self.minitagger.language = language
        self.minitagger.quiet = True

        self.minitagger.set_prediction_path(model_name)
        self.minitagger.set_model_path(model_name)

        self.display_info(data_to_use, language, model_name, feature_template, embedding_size, embedding_path)
        infos = dict()
        infos["algorithm"] = "CRF"
        infos["train_data_path"] = train_data_path
        infos["test_data_path"] = test_data_path
        infos["language"] = language
        infos["mpdel_name"] = model_name
        infos["feature_template"] =  feature_template
        infos["embedding_size"] = embedding_size
        infos["embedding_path"] = embedding_path

        # initialize feature extractor with the right feature template
        feature_extractor = FeatureExtractor_CRF_SVM(feature_template, language,
                                                         embedding_size if embedding_size else None)
        # load bitstring or embeddings data
        if feature_template == "embedding":
            feature_extractor.load_word_embeddings(embedding_path, embedding_size)
        # equip Minitagger with the appropriate feature extractor
        self.minitagger.equip_feature_extractor(feature_extractor)
        self.minitagger.extract_features(self.sequence_data, self.test_sequence,validation_sequence=self.validation_sequence, data_to_use=data_to_use)
        print("Features extracted")

        start = time.time()
        score = Score("CRF_parameter_selection", infos)

        for i in range(number_of_trial):
            mean_fscore_conll, param  = self.cv_crf(i)
            score.add_scores(mean_fscore_conll, None, None, param)
            score.save_result_to_file(infos, self.minitagger.model_path)

        score.display_results()

        # Retrained with best parameters:
        print("============================")
        print("Train with best parameters")
        print("============================")
        _, _, _, best_param, _ = score.get_max_conll_fscore()

        validation_data_path = "../../ner/nerc-conll2003/eng-simplified.testa"
        test_data_path = "../../ner/nerc-conll2003/eng-simplified.testb"
        model_name = model_name + "_finale_score"

        selection = final_training(train_data_path, validation_data_path, test_data_path, language, model_name,
                                   feature_template, embedding_size, embedding_path)
        selection.final_training("crf", best_param)

    def cv_crf(self, i):
        algorithm = "lbfgs"
        c1 = np.linspace(0,1,20*2)
        c2 = np.linspace(0,1,20*2)
        epsilon = np.logspace(-6,0,20*2)
            #ok = False
            #while not ok:
        self.minitagger.algorithm = algorithm
        self.minitagger.c1 = np.random.choice(c1)
        self.minitagger.c2 = np.random.choice(c2)
        self.minitagger.epsilon = np.random.choice(epsilon)
        self.minitagger.all_possible_state = False
        self.minitagger.all_possible_transitions = True

        parameter = dict()
        parameter["algo"] = self.minitagger.algorithm
        parameter["c1"] = self.minitagger.c1
        parameter["c2"] = self.minitagger.c2
        parameter["epsilon"] = self.minitagger.epsilon
        parameter["all_possible_states"] = self.minitagger.all_possible_state
        parameter["all_possible_transition"] = self.minitagger.all_possible_transitions

        print("Step", str(i + 1), ":")

        return self.minitagger.cross_validation(i+1, self.sequence_data, feature_template, language, embedding_path, embedding_size,  data_test=None, n_fold=5)
        #exact_score, inexact_score, conllEval = self.minitagger.train(self.sequence_data, self.test_sequence,
         #                                                        feature_already_extracted=True, id=i)


    @staticmethod    
    def display_results(conll_fscore, conll_precision, conll_recall, conll_parameters):
        print()
        print()
        print("=================================")
        print("Results")
        print("=================================")
        print("F1score", conll_fscore)
        #print("Std", conll_stds)
        print()
        print("=================================")
        print("Best results")
        print("=================================")
        argmax = np.argmax(conll_fscore)
        print(" Best conll f1score:", conll_fscore[argmax])
        print(" Corresponding conll precision:", conll_precision[argmax])
        print(" Corresponding conll recall:", conll_recall[argmax])
        print(" Corresponding id:", argmax)
        #print(" Correspondind std:", conll_stds[argmax])
        print(" Corresponding parameters:", conll_parameters[argmax])
   
    @staticmethod    
    def display_info(number_of_sentence, language, model_name, feature_template, embedding_size=None,embedding_path=None):
        print("Number of sentences: ", number_of_sentence)
        print("Language: ", language)
        print("Model name: ", model_name)
        print("Feature template:", feature_template)
        if feature_template == "embedding":
            print("Embedding size: ", embedding_size)

        
    def save_to_file(self, conll_fscore, conll_precision, conll_recall, conll_parameters, infos, model_name):
        argmax = np.argmax(conll_fscore)
        with open(os.path.join(self.minitagger.model_path, "best_conll_param.json"), "w") as file: # TODO change path => put everything in model folder
            conll = dict()
            conll["fscore"] = conll_fscore[argmax]
            conll["precision"] = conll_precision[argmax]
            conll["recall"] = conll_recall[argmax]
            #conll["std"] = conll_stds[argmax]
            conll["parameter"] = conll_parameters[argmax]
            conll["list_fscore"] = conll_fscore
            conll["list_precision"] = conll_precision
            conll["list_recall"] = conll_recall
            #conll["list_std"] = conll_stds
            conll["list_parameter"] = conll_parameters

            json_data = {"infos": infos, "results": conll}
            json.dump(json_data, file,indent=2, separators=(',', ': '))

if __name__ == "__main__":

    a = "test"
    language = "en"
    embedding_size = 300
    if a == "test":
        train_data_path = "../../ner/small_datasets/eng-simplified.train"
        validation_data_path = "../../ner/small_datasets/eng-simplified.testa"
        test_data_path = "../../ner/small_datasets/eng-simplified.testb"

        #train_data_path = "../../ner/nerc-conll2003/eng-simplified.train"
        #test_data_path = "../../ner/nerc-conll2003/eng-simplified.testa"
        feature_template = "baseline"
        embedding_path = "../../word_embeddings/glove"
        model_name = "_test_baseline_parameter_selection_2"
        number_of_trial = 2

    elif a == "wikiner":
        train_data_path = "../../wikiner_dataset/aij-wikiner-en-wp2-simplified"
        validation_data_path = "../../ner/nerc-conll2003/eng-simplified.testa"
        test_data_path = "../../ner/nerc-conll2003/eng-simplified.train"
        feature_template = "embedding"
        embedding_path = "../../word_embeddings/glove"
        model_name = "_wikiner_emb" + str(embedding_size) + "_parameter_selection_2"
        number_of_trial = 10
    elif a =="conll":
        train_data_path = "../../ner/nerc-conll2003/eng-simplified.train"
        validation_data_path = None
        test_data_path = "../../ner/nerc-conll2003/eng-simplified.testa"
        feature_template = "embedding"
        embedding_path = "../../word_embeddings/glove"
        model_name = "_conll_emb" + str(embedding_size) + "_parameter_selection_1"
        number_of_trial = 10



    selection = Parameter_selection()


    # elif algorithm == "CRF":
    #selection.parameter_selection_crf(train_data_path,validation_data_path, test_data_path,
    #                                 language, "CRF"+model_name, feature_template,
    #                                 embedding_size,embedding_path, number_of_trial = number_of_trial)

    #if algorithm == "SVM":
    selection.parameter_selection_svm(train_data_path,test_data_path,
                                      language, "SVM"+model_name, feature_template,
                                      embedding_size,embedding_path, number_of_trial=number_of_trial)
