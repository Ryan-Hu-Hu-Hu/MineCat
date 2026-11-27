"""EC2Vec enzyme-function embeddings (1,024-d) looked up by EC number.

EC2Vec (GPL-3.0) is not redistributed here: clone https://github.com/MengLiu90/EC2Vec and
point MINECAT_EC2VEC at its Embedding_Results/embedded_EC_number.csv (first column = EC
number, remaining 1,024 columns = embedding). Unknown EC numbers map to a zero vector.
"""
import csv

import numpy as np

from .. import config

EC2VEC_DIM = 1024


def load_ec2vec_table(path=None):
    table = {}
    with open(path or config.EC2VEC_CSV, newline="") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            table[row[0].strip()] = np.asarray(row[1:], dtype=np.float32)
    return table


def ec2vec(ec_numbers, table=None):
    """(N, 1024) float32 array for a list of EC strings (e.g. '1.1.1.1')."""
    table = table or load_ec2vec_table()
    zero = np.zeros(EC2VEC_DIM, dtype=np.float32)
    return np.stack([table.get(str(ec).strip(), zero) for ec in ec_numbers]).astype(np.float32)
