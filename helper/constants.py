from enum import Enum
class dataset_type(Enum):
    conll, wikiner, new_dataset = range(3)

class embedding_type(Enum):
    fasttext,polyglot = range(2)


MAPPING_LANGUAGE = {
    #german
    "de" : "de",
    "als" : "de",
    "lb" : "de",
    "nds" : "de",
    "ksh" : "de",
    "pfl" : "de",
    "pdc" : "de",
    #italian
    "it" : "it",
    "pms" : "it",
    "lmo" : "it",
    "scn" : "it",
    "vec" : "it",
    "nap" : "it",
    "sc" : "it",
    "co" : "it",
    "rm" : "it",
    "lij" : "it",
    "fur" : "it",
    #french
    "fr" : "fr",
    "oc" : "fr",
    "ca" : "fr",
    "ht" : "fr",
    "wa" : "fr",
    "nrm" : "fr",
    "pcd" : "fr",
    "frp" : "fr"
}

it_dialect = ["pms", "lmo", "scn", "vec", "nap", "sc", "co", "rm", "lij", "fur"]
