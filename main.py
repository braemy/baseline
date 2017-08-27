import argparse
import json
import os
import sys

import yaml

import helper.utils as utils

from FeatureExtractor_CRF_SVM import FeatureExtractor_CRF_SVM
# from MinitaggerCRF import MinitaggerCRF
from MinitaggerCRF import MinitaggerCRF
from MinitaggerSVM import MinitaggerSVM
from helper.SequenceData import SequenceData
from helper.model_evaluation import report_fscore_from_file

# Used for instances without gold labels
ABSENT_GOLD_LABEL = "<NO_GOLD_LABEL>"


def main(exp):
    with open(os.path.join('.', 'experiments.yml'), 'r') as ymlfile:
        experiments = yaml.load(ymlfile)

    for elem in experiments['experiments'][exp]:
        trained_model = elem[0]
        test = elem[1]
        print("======================")
        print("Train on {0}, test {1}".format(trained_model, test))
        print("======================")

        train_data_path = experiments['datasets'][test]
        print(train_data_path)
        pos_tag = True
        language = experiments['language']
        model_name = experiments['models'][trained_model]
        train = False
        prediction_name = "{0}_on_{1}".format(trained_model, test)
        wikiner = False

        if "SVM" in model_name:
            minitagger = MinitaggerSVM()
        else:
            minitagger = MinitaggerCRF()
        sequence_data = SequenceData(train_data_path,pos_tag, language=language)

        minitagger.language = language
        minitagger.set_model_path(model_name)
        if train:
            minitagger.set_prediction_path(model_name)
        else:
            print(model_name)
            minitagger.set_prediction_path(model_name+"_"+prediction_name)

        if wikiner:
            minitagger.wikiner = True

        if train:

            # initialize feature extractor with the right feature template
            feature_extractor = FeatureExtractor_CRF_SVM(args.feature_template, args.language,
                                                         args.embedding_size if args.embedding_size else None)
            # load bitstring or embeddings data
            feature_extractor.morphological_features = "regular"
            feature_extractor.token_features2 = True
            feature_extractor.token_features1 = True
            feature_extractor.feature_template = "baseline"



            # equip Minitagger with the appropriate feature extractor
            minitagger.equip_feature_extractor(feature_extractor)
            test_data = SequenceData(args.test_data_path, args.pos_tag) if args.test_data_path else None
            if test_data is not None:
                # Test data should be fully labeled
                assert (not test_data.is_partially_labeled), "Test data should be fully labeled"
            minitagger.debug = args.debug
            if minitagger.debug:
                assert args.prediction_path, "Path for prediction should be specified"

            if feature_extractor.feature_template == "embedding":
                vocabulary = sequence_data.vocabulary.union(test_data.vocabulary)
                feature_extractor.load_word_embeddings(args.embedding_path, args.embedding_size, vocabulary)

            # minitagger.feature_extractor.all_features = False
            minitagger.train(sequence_data, test_data)
            #    minitagger.cross_validation(sequence_data, test_data)
            # minitagger.train_sparse(sequence_data, test_data)
            #   minitagger.save(args.model_path)

            # minitagger.cross_validation(sequence_data, test_data, 5)
        # predict labels in the given data.
        else:
            feature_template = experiments['parameters']['feature_template']
            embedding_size = experiments['parameters']['embedding_size']
            embedding_path = experiments['parameters']['embedding_path']


            feature_extractor = FeatureExtractor_CRF_SVM(feature_template, language,
                                                         embedding_size if embedding_size else None)
            if feature_template == "embedding":
                feature_extractor.load_word_embeddings(embedding_path, embedding_size, sequence_data.vocabulary)

            minitagger.equip_feature_extractor(feature_extractor)
            if "CRF" in model_name:
                minitagger.extract_features(None, sequence_data)
                minitagger.load_model()
                pred_labels = minitagger.crf.predict(minitagger.test_features)
                minitagger.test_sequence.save_prediction_to_file(pred_labels, minitagger.prediction_path)
                exact_score, inexact_score, conllEval = report_fscore_from_file(
                    os.path.join(minitagger.prediction_path, "predictions.txt"),
                    wikiner=minitagger.wikiner, quiet=True)

                with open(os.path.join(minitagger.prediction_path,"results.txt"), "w") as file:
                    file.write("CoNLL Eval\n")
                    file.write(json.dumps(conllEval))
                    file.write("\n\n")

                    file.write("CoNLL Eval\n")
                    file.write(json.dumps(exact_score))
                    file.write("\n\n")

                    file.write("CoNLL Eval\n")
                    file.write(json.dumps(inexact_score))
                    file.write("\n\n")

                minitagger.display_results("Conll", conllEval)
                minitagger.display_results("Exact", exact_score)
                minitagger.display_results("Inexact", inexact_score)

            else:
                minitagger.load(minitagger.model_path)
                minitagger.set_is_training(False)
                minitagger.extract_features(None, sequence_data)
                pred_labels, _ = minitagger.predict()
                report_fscore_from_file(os.path.join(minitagger.prediction_path, "predictions.txt"), quiet=False)



            # optional prediction output
            # write predictions to file
            # if args.prediction_path:
            #     file_name = os.path.join(args.project_dir, args.prediction_path, "predictions.txt")
            #     with open(file_name, "w") as outfile:
            #         label_index = 0
            #         for sequence_num, (word_sequence, label_sequence) in enumerate(sequence_data.sequence_pairs):
            #             for position, word in enumerate(word_sequence):
            #                 if not label_sequence[position] is None:
            #                     gold_label = label_sequence[position]
            #                 else:
            #                     gold_label = ABSENT_GOLD_LABEL
            #                 outfile.write(word + "\t" + gold_label + "\t" + pred_labels[label_index] + "\n")
            #                 label_index += 1
            #             if sequence_num < len(sequence_data.sequence_pairs) - 1:
            #                 outfile.write("\n")


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    # argparser.add_argument("--train_data_path", type=str, help="path to data (used for training/testing)",
    #                        required=False)
    # argparser.add_argument("--model_name", type=str, help="name used to store the model and the predictions",
    #                        required=False, default='svm')
    # argparser.add_argument("--train", action="store_true", help="train the tagger on the given data")
    # argparser.add_argument("--feature_template", type=str, default="baseline",
    #                        help="feature template (default: %(default)s)")
    # argparser.add_argument("--embedding_path", type=str, help="path the folder containing word embeddings")
    # argparser.add_argument("--embedding_size", type=int, choices=[50, 100, 200, 300],
    #                        help="the size of the word embedding vectors")
    # argparser.add_argument("--quiet", action="store_true", help="no messages")
    # argparser.add_argument("--test_data_path", type=str, help="path to test data set (used for training)")
    # argparser.add_argument("--language", type=str, choices=["en", "de", "fr", "it"],
    #                        help="language of the data set [en, de,fr]", required=True)
    # argparser.add_argument("--debug", action="store_true", help="produce some files for debugging")
    # argparser.add_argument("--pos_tag", action="store_true",
    #                        help="indicate if the part-of-speech tag is present or not")
    # argparser.add_argument("--project_dir", type=str, help="directory of the path")
    # argparser.add_argument("--prediction_name", type=str, help="name of the prediction")
    # argparser.add_argument("--wikiner", action="store_true",
    #                        help="if we are using wikiner dataset, use this arg to use appropriate scoring function")

    argparser.add_argument("--experiment_set",nargs='+',  type=str, help="")


    parsed_args = argparser.parse_args()
    for exp in parsed_args.experiment_set:
        main(exp)
