# _*_ coding:utf-8 _*_
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from utils import *
import tensorly
import json
import os
import faiss
from TFP import TripleFeaturePropagation

import scipy.sparse as sp

seed = 123456
np.random.seed(seed)

# choose the GPU, "-1" represents using the CPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
tensorly.set_backend('tensorflow')

gpus = tf.config.experimental.list_physical_devices(device_type="GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# choose the base model and dataset
model = ["Dual_AMN", "RSN", "AlignE", "MRAEA", "TransEdge", "RREA"][3]
dataset = ["DBP_ZH_EN/", "DBP_JA_EN/", "DBP_FR_EN/", "SRPRS_FR_EN/", "SRPRS_DE_EN/"][3]

if "DBP" in dataset:
    path = "./EA_datasets/" + ("sharing/" if model == "TransEdge" else "mapping/") + dataset + "0_3/"
else:
    path = "./EA_datasets/" + ("sharing/" if model == "TransEdge" else "mapping/") + dataset

train_pair, test_pair = load_aligned_pair(path)

# obtain the literal features of entities, only work on DBP15K & SRPRS
# for the first run, you need to download the pre-train word embeddings from "http://nlp.stanford.edu/data/glove.6B.zip"
# unzip this file and put "glove.6B.300d.txt" into the root of TFP

ent_emb = load_pre_features(dataset,"./glove.6B.300d.txt",mode = "hybrid-level")
ent_emb = tf.cast(ent_emb, "float32")

# decoding algorithm
# Triple Feature Propagation based on the entity embedding
node_size, rel_size, ent_tuple, triples_idx, ent_ent, ent_ent_val, rel_ent, ent_rel = load_graph(path)

print("Begin to Triple Feature Propagate:")
Triple_FP = TripleFeaturePropagation(train_pair, ent_emb)
features = Triple_FP.propagation(node_size, rel_size, ent_tuple, triples_idx, ent_ent, ent_ent_val, rel_ent, ent_rel)

sims = cal_sims(test_pair,features)
sims = tf.exp(sims/0.02)

for k in range(15):
    sims = sims / tf.reduce_sum(sims,axis=1,keepdims=True)
    sims = sims / tf.reduce_sum(sims,axis=0,keepdims=True)
test(sims,"sinkhorn")

# the results of base model
csls_sims(test_pair,ent_emb)
