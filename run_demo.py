import os
import sys
import argparse
from langdetect import detect
import main as minitagger_main

CONFIG_PATH = '/Users/baptisteraemy/GoogleDrive/Swisscom/swisscom-demo/server/'

sys.path.append(os.path.abspath(CONFIG_PATH))
import config


def parse_input(input_sentence, input_path):
    f = open(input_path, "w")
    for token in input_sentence.split():
        s = token.strip() + "\n"
        f.write(s)
    f.close()


def parse_output(output_path):
    f = open(config.paths['ner'])
    pred_labels = []
    for line in f:
        pred_label = line.split()[2]
        pred_labels.append(pred_label)
    return " ".join(pred_labels)

def detect_language(sentence):
    """
    detect the langage among FR, ENG and GER, if it's none of them, we assume that the langage was english
    :param sentence: sentence to detect the langage from
    :type sentence: str
    :return: en, fr, de
    :rtype: str
    """
    language = detect(sentence)
    supported_langage = ['fr', 'en', 'de']
    return language if language in supported_langage else 'en'


def main(args):
    sentence = args.sentence
    sentence = sentence.strip('"')
    print(sentence)
    language = detect_language(sentence)
    input_path = config.paths['ner'] +"server_test_data.txt"
    output_path = config.paths['ner'] +"predictions.txt"
    model_path = "model_path_wikiner" + ("_embedding_" + str(50) if args.model == "embedding" else "")
    prediction_path = "predictions"
    program = config.paths['ner'] + "main.py"

    parse_input(sentence, input_path)
    command = "python " + program + " --data_path " + input_path + " --model_path " + model_path + " --prediction_path " \
              + prediction_path + " --project_dir " + config.paths['ner']+" --language " + language + " > /dev/null"  # TODO CHANGE HARD CODING LANGUAGE => put language
    os.system(command)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--sentence", type=str, help="sentence for prediction", required=True)
    argparser.add_argument("--model", type=str, help="model to use", required=True)

    parsed_args = argparser.parse_args()
    main(parsed_args)
