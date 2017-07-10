import json
import time
import numpy as np

from FeatureExtractor_CRF_SVM import FeatureExtractor_CRF_SVM
from MinitaggerCRF import MinitaggerCRF
from MinitaggerSVM import MinitaggerSVM
from helper.score import Score
from final_training import Final_training
from helper.SequenceData import SequenceData
from helper.utils import *

class Parameter_selection(object):
    def __init__(self):
        self.minitagger = None

    # def parameter_selection_svm(self, train_data_path, test_data_path, language, model_name, feature_template,
    # embedding_size=None, embedding_path=None, number_of_trial=20, seed=123456):
    def parameter_selection_svm(self, parameters):
        train_data_path = parameters["train_data_path"]
        validation_data_path = parameters["validation_data_path"]
        test_data_path = parameters["test_data_path"]
        language = parameters["language"]
        feature_template = parameters["feature_template"]
        if feature_template == "embedding":
            embedding_size = parameters["embedding_size"]
            embedding_path = parameters["embedding_path"]
        model_name = parameters["model_name"]
        number_of_trial = parameters["number_of_trial"] if "number_of_trial" in parameters else 5

        minitagger = MinitaggerSVM()

        sequence_data = SequenceData(train_data_path, language=language, pos_tag=True)
        test_sequence = SequenceData(test_data_path, language=language, pos_tag=True)
        minitagger.language = language
        minitagger.set_prediction_path(model_name)
        minitagger.set_model_path(model_name)
        # list of parameter:
        epsilon = np.logspace(-6, 0, number_of_trial * 3)
        cost = np.logspace(-5, 4, number_of_trial * 3)

        self.display_info(len(sequence_data.sequence_pairs), parameters)

        parameter_aready_tested = []

        # initialize feature extractor with the right feature template
        feature_extractor = FeatureExtractor_CRF_SVM(feature_template, language,
                                                         embedding_size if feature_template=="embedding" else None)
        # load bitstring or embeddings data
        if feature_template == "embedding":
            feature_extractor.load_word_embeddings(embedding_path, embedding_size)
        # equip Minitagger with the appropriate feature extractor
        minitagger.equip_feature_extractor(feature_extractor)
        minitagger.extract_features(sequence_data, test_sequence)

        score = Score("SVM_parameter_selection", parameters)

        for i in range(number_of_trial):

            ok = False
            while not ok:
                parameter = dict()
                minitagger.epsilon = np.random.choice(epsilon)
                minitagger.cost = np.random.choice(cost)
                parameter["epsilon"] = minitagger.epsilon
                if parameter not in parameter_aready_tested:
                    parameter_aready_tested.append(parameter)
                    ok = True

            mean_fscore_conll, param = minitagger.cross_validation(i+1,
                                                                   sequence_data,
                                                                   feature_template,
                                                                   language,
                                                                   embedding_path if feature_template == "embedding" else None,
                                                                   embedding_size if feature_template == "embedding" else None,
                                                                   data_test=None,
                                                                   n_fold=5)
            score.add_scores(mean_fscore_conll, None, None, param)
            score.save_result_to_file(minitagger.model_path)

        score.display_results()

        # Retrained with best parameters:
        print("============================")
        print("Trained with best parameters")
        print("============================")
        _, _, _, best_param, _ = score.get_max_conll_fscore()

        validation_data_path = None
        parameters["model_name"] = model_name + "_finale_score"
        parameters["method"] = "SVM"
        parameters["best_param"] = best_param
        selection = Final_training(parameters)
        selection.train()

    def parameter_selection_crf(self, parameters):
        train_data_path = parameters["train_data_path"]
        validation_data_path = parameters["validation_data_path"]
        test_data_path = parameters["test_data_path"]
        language = parameters["language"]
        feature_template = parameters["feature_template"]
        if feature_template == "embedding":
            embedding_size = parameters["embedding_size"]
            embedding_path = parameters["embedding_path"]
        model_name = parameters["model_name"]
        number_of_trial = parameters["number_of_trial"] if "number_of_trial" in parameters else 5

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

        self.display_info(data_to_use, parameters)


        # initialize feature extractor with the right feature template
        feature_extractor = FeatureExtractor_CRF_SVM(feature_template, language,
                                                     embedding_size if feature_template == "embedding" else None)
        # load bitstring or embeddings data
        if feature_template == "embedding":
            feature_extractor.load_word_embeddings(embedding_path, embedding_size)
        # equip Minitagger with the appropriate feature extractor
        self.minitagger.equip_feature_extractor(feature_extractor)
        self.minitagger.extract_features(self.sequence_data, self.test_sequence,
                                         validation_sequence=self.validation_sequence, data_to_use=data_to_use)
        print("Features extracted")

        start = time.time()
        score = Score("CRF_parameter_selection", parameters)

        for i in range(number_of_trial):
            mean_fscore_conll, param = self.cv_crf(i)
            score.add_scores(mean_fscore_conll, None, None, param)
            score.save_result_to_file(self.minitagger.model_path)

        score.display_results()

        # Retrained with best parameters:
        print("============================")
        print("Train with best parameters")
        print("============================")
        _, _, _, best_param, _ = score.get_max_conll_fscore()

        parameters["model_name"] = model_name + "_finale_score"

        parameters["best_param"] = best_param
        parameters["method"] = "CRF"
        selection = Final_training(parameters)
        selection.train()

    def cv_crf(self, i):
        algorithm = "lbfgs"
        c1 = np.linspace(0, 1, 20 * 2)
        c2 = np.linspace(0, 1, 20 * 2)
        epsilon = np.logspace(-6, 0, 20 * 2)
        # ok = False
        # while not ok:
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

        # return self.minitagger.cross_validation(i + 1,
        #                                         self.sequence_data,
        #                                         feature_template, language, embedding_path,
        #                                         embedding_size, data_test=None, n_fold=5)
        # exact_score, inexact_score, conllEval = self.minitagger.train(self.sequence_data, self.test_sequence,
        #                                                        feature_already_extracted=True, id=i)

    @staticmethod
    def display_results(conll_fscore, conll_precision, conll_recall, conll_parameters):
        print()
        print()
        print("=================================")
        print("Results")
        print("=================================")
        print("F1score", conll_fscore)
        print()
        print("=================================")
        print("Best results")
        print("=================================")
        argmax = np.argmax(conll_fscore)
        print(" Best conll f1score:", conll_fscore[argmax])
        print(" Corresponding conll precision:", conll_precision[argmax])
        print(" Corresponding conll recall:", conll_recall[argmax])
        print(" Corresponding id:", argmax)
        print(" Corresponding parameters:", conll_parameters[argmax])
   
    @staticmethod    
    def display_info(number_of_sentence, parameters):
        print("Number of sentences: ", number_of_sentence)
        print("Language: ", parameters["language"])
        print("Model name: ", parameters["model_name"])
        print("Feature template:", parameters["feature_template"])
        if parameters["feature_template"] == "embedding":
            print("Embedding size: ", parameters["embedding_size"])

    def save_to_file(self, conll_fscore, conll_precision, conll_recall, conll_parameters, infos, model_name):
        argmax = np.argmax(conll_fscore)
        with open(os.path.join(self.minitagger.model_path, "best_conll_param.json"),
                  "w") as file:  # TODO change path => put everything in model folder
            conll = dict()
            conll["fscore"] = conll_fscore[argmax]
            conll["precision"] = conll_precision[argmax]
            conll["recall"] = conll_recall[argmax]
            conll["parameter"] = conll_parameters[argmax]
            conll["list_fscore"] = conll_fscore
            conll["list_precision"] = conll_precision
            conll["list_recall"] = conll_recall
            conll["list_parameter"] = conll_parameters

            json_data = {"infos": infos, "results": conll}
            json.dump(json_data, file, indent=2, separators=(',', ': '))


if __name__ == "__main__":
    selection = Parameter_selection()
    parameters = load_parameters("parameter_selection")
    selection.parameter_selection_svm(parameters)
    print ()
