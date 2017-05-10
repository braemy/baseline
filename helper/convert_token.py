import pickle

import numpy as np
from tqdm import tqdm

from helper.SequenceData import SequenceData


sequence = SequenceData("../../wikiner_dataset/aij-wikiner-en-wp2-simplified", pos_tag=True)

print("Loading embedding...")
file_name = "../../word_embeddings/en/vocab_word_embeddings_50.p"
file_name = "../../word_embeddings/en/word_embeddings_dict_50.p"
with open(file_name, "rb") as file:
    vocab = pickle.load(file)

output = []
dummy = len(vocab)
for i,(sentence, *_) in tqdm(enumerate(sequence.sequence_pairs)):
    output.append(list(map(lambda x:vocab.get(x.lower(), dummy),sentence)))
