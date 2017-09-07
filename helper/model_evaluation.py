from helper.conlleval import evaluate, report

"""
Functions used to compute the exact Fscore, Ineact F1score and ConllFesocre
The main function to use is the "report_fscore_from_file" that will apply all the fscore on the given file
"""

def estimate_inexact_fscore(y_true, y_pred, b_equals_i=False):
    label_list = ["B-PER", "I-PER", "B-LOC", "I-LOC", "B-ORG", "I-ORG", "B-MISC", "I-MISC"]

    tp = 0
    tn = 0
    fp = 0
    fn = 0

    for i in range(len(y_pred)):
        if b_equals_i:
            if y_true[i] == "O":
                if y_pred[i] == "O":
                    tn += 1
                else:
                    # prediction is in the list of label
                    fp += 1

            else:
                assert "-" in y_pred[i] and "-" in y_true[i], "true label " + y_true[i] + " or predicted label " + \
                                                              y_pred[i] + " should contains '-'"
                pred_b_i, pred_class = y_pred[i].split("-")
                true_b_i, true_class = y_true[i].split("-")

                if true_class == pred_class:
                    tp += 1
                else:
                    fp += 1  # TODO check that part with Claudiu, that case didn't exist in Thanos part
        else:
            if y_true[i] == "O":
                if y_pred[i] == "O":
                    tn += 1
                else:
                    # prediction is in the list of label
                    fp += 1

            # True label is not O => it's an entity
            elif y_true[i] == y_pred[i]:
                tp += 1
            elif y_pred[i] in label_list:
                fp += 1
            else:
                fn += 1

    return tp, tn, fp, fn


def estimate_exact_fscore(y_true, y_pred):
    pairs = []
    start = False
    end = False

    tp = 0
    tn = 0
    fp = 0
    fn = 0
    total_seq = 0

    for i in range(len(y_pred)):
        if y_true[i] == "O":
            if y_pred[i] == "O":
                tn += 1
            else:
                fp += 1
        else:
            y_true[i]
            if y_true[i][0] != 'B' and y_true[i][0] != 'I':
                y_true[i] = 'O'
                true_b_i, true_class = 'O', 'O'
            else:
                # assert "-" in y_true[i], "true label " + y_true[i] + " should contains '-'"
                true_b_i, true_class = y_true[i].split("-")

            if true_b_i == "B":
                if start:  # in case B-PER I-PER B-LOC I-LOC => is sequence started, need to add it when we are at B-LOC
                    end_index = i
                    start = False
                    pairs.append((start_index, end_index))
                if i < len(y_true) - 1:  # check that we are not at the end of the sentence

                    # if y_true[i+1] != "O" and y_true[i+1].split("-")[1] != true_class:
                    #    print(y_true)
                    #    print(y_pred)
                    #    print("should not happen: i" + y_true[i] + " i+1:" + y_true[i+1])

                    if y_true[i + 1] == "O" or y_true[i + 1].split("-")[
                        0] == "B":  # if next label start with "B" => not a sequence
                        start_index = i
                        pairs.append((start_index,))
                        continue

                else:  # end of the sentence
                    if y_true[i] == "B":
                        start_index = i
                        pairs.append((start_index,))
                        continue

        if y_true[i] != "O" and true_b_i == "B" and (not start):
            start = True
            total_seq += 1
            start_index = i
        # TODO only condition to stop or also if B-PER and I-LOC
        elif start and ((y_true[i] == "O") or i == len(y_true) - 1):
            start = False
            if i == len(y_true) - 1:
                end_index = i + 1
                pairs.append((start_index, end_index))
            else:
                end_index = i
                pairs.append((start_index, end_index))
    for pair in pairs:
        if len(pair) == 1:
            if y_true[pair[0]] == y_pred[pair[0]]:
                tp += 1
            else:
                fn += 1
        if len(pair) == 2:
            if y_true[pair[0]:pair[1]] == y_pred[pair[0]:pair[1]]:
                tp += 1
            else:
                fn += 1

    return tp, tn, fp, fn, total_seq

def estimate_precision(tp, fp):
    if tp == 0: return 0
    return float(tp) / (tp + fp) * 100


def estimate_recall(tp, fn):
    if tp == 0: return 0
    return float(tp) / (tp + fn) * 100


def estimate_fscore(precision, recall):
    if precision == 0 or recall == 0: return 0
    return float(2 * precision * recall) / (precision + recall)


def split_prediction_true_label(file_name):
    f = open(file_name, "r", encoding='utf-8')

    sentence = []
    y_true = []
    y_pred = []

    true_sequence = []
    pred_sequence = []

    for line in f:
        tokens = line.split()
        if len(tokens) != 0:
            sentence.append(tokens[0])
            y_true.append(tokens[1])
            y_pred.append(tokens[2])
        else:
            true_sequence.append(y_true)
            pred_sequence.append(y_pred)
            sentence = []
            y_true = []
            y_pred = []

    if sentence:
        true_sequence.append(y_true)
        pred_sequence.append(y_pred)

    return true_sequence, pred_sequence


def report_fscore_from_file(prediction_file, quiet=True, output_conll_file=None):
    """
    report the fscore for the given file
    :param prediction_file: file containing the prediction with format:  TOKEN GOLD_LABEL PREDICTED_LABEL
    :param quiet: False to activate the print
    :param output_conll_file: file to save the result
    :return: exact_score, inexact_score and conll_score where each one is a dict with recall, precision and f1-score
    """
    print(prediction_file)
    true_label, pred_label = split_prediction_true_label(prediction_file)
    with open(prediction_file, encoding='utf-8') as f:
        counts = evaluate(f, None)
    conllEval = report(counts, output_conll_file)
    print(conllEval)
    exact_score, inexact_score = report_fscore(true_label, pred_label, wikiner, quiet)

    return exact_score, inexact_score, conllEval

def report_fscore(true_label, pred_label, quiet=True):
    """
    compute the exact and inexact score
    :param true_label: gold label
    :param pred_label: predicted label
    :param quiet: False to print intermediate results
    :return: return exact and inexact score with precision, recall and f1-score
    """
    tp_exact = 0
    tn_exact = 0
    fp_exact = 0
    fn_exact = 0

    total_seq = 0

    tp_inexact = 0
    tn_inexact = 0
    fp_inexact = 0
    fn_inexact = 0

    for y_true, y_pred in zip(true_label, pred_label):

        tp, tn, fp, fn = estimate_inexact_fscore(y_true, y_pred)
        tp_inexact += tp
        tn_inexact += tn
        fp_inexact += fp
        fn_inexact += fn

        tp, tn, fp, fn, seq = estimate_exact_fscore(y_true, y_pred)
        tp_exact += tp
        tn_exact += tn
        fp_exact += fp
        fn_exact += fn
        total_seq += seq

    exact_score = dict()
    precision = estimate_precision(tp_exact, fp_exact)
    exact_score["precision"] = precision
    recall = estimate_recall(tp_exact, fn_exact)
    exact_score["recall"] = recall
    e_fscore = estimate_fscore(precision, recall)
    exact_score["f1score"] = e_fscore

    inexact_score = dict()
    precision = estimate_precision(tp_inexact, fp_inexact)
    inexact_score["precision"] = precision
    recall = estimate_recall(tp_inexact, fn_inexact)
    inexact_score["recall"] = recall
    i_fscore = estimate_fscore(precision, recall)
    inexact_score["f1score"] = i_fscore

    if not quiet:
        print("Exact fscore: ", "{0:.3f}".format(e_fscore))
        print("Exact precision: ", "{0:.3f}".format(precision))
        print("Exact recall: ", "{0:.3f}".format(recall))

        print()
        print("Inexact fscore: ", "{0:.3f}".format(i_fscore))
        print("Inexact precision: ", "{0:.3f}".format(precision))
        print("Inexact recall: ", "{0:.3f}".format(recall))

        print()

    return exact_score, inexact_score
