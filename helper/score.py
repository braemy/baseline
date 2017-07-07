import json
import os
import pickle

import numpy as np


class Score(object):
    def __init__(self, name, infos):
        self.infos = infos
        self.name = name
        self.fscore_conll = []
        self.precision_conll = []
        self.recall_conll = []

        self.fscore_exact = []
        self.precision_exact = []
        self.recall_exact = []

        self.fscore_inexact = []
        self.precision_inexact = []
        self.recall_inexact = []

        self.parameters = []

        self.iterations = []
        self.time_step = []

    def add_conll(self, score):
        f, p, r = self.split_score(score)
        self.fscore_conll.append(f)
        self.precision_conll.append(p)
        self.recall_conll.append(r)

    def add_exact(self, score):
        f, p, r = self.split_score(score)
        self.fscore_exact.append(f)
        self.precision_exact.append(p)
        self.recall_exact.append(r)

    def add_inexact(self, score):
        f, p, r = self.split_score(score)
        self.fscore_inexact.append(f)
        self.precision_inexact.append(p)
        self.recall_inexact.append(r)

    def add_iteration(self, iteration):
        self.iterations.append(iteration)

    def add_time_step(self, time):
        self.time_step.append(time)

    def add_parameters(self, param=None):
        self.parameters.append(param)

    def get_last_parameters(self):
        return self.parameters[-1]

    def add_new_iteration(self, iteration, time, conll, exact, inexact, param):
        self.add_iteration(iteration)
        self.add_time_step(time)
        self.add_conll(conll)
        self.add_exact(exact)
        self.add_inexact(inexact)
        self.add_parameters(param)

    def save_class_to_file(self, path):
        with open(os.path.join(path, self.name + ".p"), 'wb') as fid:
            pickle.dump(self, fid)

    def add_scores(self, conll, exact, inexact, param=None):
        self.add_conll(conll)
        self.add_exact(exact)
        self.add_inexact(inexact)
        self.add_parameters(param)

    def get_max_conll_fscore(self):
        argmax = np.argmax(self.fscore_conll)
        return self.fscore_conll[argmax], self.precision_conll[argmax], self.recall_conll[argmax], self.parameters[
            argmax], argmax

    def get_mean_conll_fscore(self):
        return {"f1score": np.mean(self.fscore_conll),
                "precision": np.mean(self.precision_conll),
                "recall": np.mean(self.recall_conll)}

    @staticmethod
    def split_score(score):
        if score is None:
            return None, None, None
        return score["f1score"], score["precision"], score["recall"]

    def display_results(self):
        fscore_c, precision_c, recall_c, param, argmax = self.get_max_conll_fscore()
        print()
        print()
        print("=================================")
        print("Results")
        print("=================================")
        print("F1score", self.fscore_conll)
        print("Precision", self.precision_conll)
        print("Recall", self.recall_conll)
        print()
        print("=================================")
        print("Best results")
        print("=================================")
        print(" Best conll f1score:", fscore_c)
        print(" Correspondind precision:", precision_c)
        print(" Correspondind recall:", recall_c)
        print(" Corresponding parameters:", param)

    def save_result_to_file(self, infos, dir_output):

        fscore_c, precision_c, recall_c, param_c, argmax = self.get_max_conll_fscore()
        with open(os.path.join(dir_output, "best_conll_param.json"), "w") as file:
            conll = dict()
            conll["fscore"] = fscore_c
            conll["precision"] = precision_c
            conll["recall"] = recall_c
            conll["parameter"] = param_c
            conll["list_fscore"] = self.fscore_conll
            conll["list_precision"] = self.precision_conll
            conll["list_recall"] = self.recall_conll
            conll["list_parameter"] = self.parameters

            json_data = {"infos": infos, "results": conll}
            # pickle_data(json_data, dir_output, "best_conll_param.json")
            json.dump(json_data, file, indent=2, separators=(',', ': '))

    @staticmethod
    def get_best_parameter(list_score):
        best_conll_mean = -1
        best_param = None
        for score in list_score:
            mean_conll = score.get_mean_conll_fscore()
            if mean_conll >= best_conll_mean:
                best_conll_mean = mean_conll
                best_param = score.get_last_parameters()
        return best_conll_mean, best_param
