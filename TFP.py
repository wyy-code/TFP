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
import time

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

if model == "RSN":
    emb_path = "Embeddings/RSN/%s" % dataset
    ent_emb = tf.cast(np.load(emb_path + "ent_emb.npy"), "float32")
    ent_dic, rel_dic = json.load(open(emb_path + "ent_id2id.json")), json.load(open(emb_path + "rel_id2id.json"))
    sort_key = []
    sort_value = []
    for i in range(ent_emb.shape[0]):
        sort_key.append(i)
    for i in sort_key:
        sort_value.append(int(ent_dic[str(i)]))

    ent_emb = ent_emb.numpy()[sort_value]
    ent_emb = tf.cast(ent_emb, "float32")
    print("RSN")
elif model == "AlignE":
    ent_emb = tf.cast(np.load("Embeddings/AlignE/%sent_emb.npy" % dataset), "float32")
    print("AlignE")

elif model == "Dual_AMN":
    ent_emb = tf.cast(np.load("Embeddings/Dual_AMN/%sent_emb.npy" % dataset), "float32")
    print("Dual_AMN")

elif model == "MRAEA":
    ent_emb = tf.cast(np.load("Embeddings/MRAEA/%sent_emb.npy" % dataset), "float32")
    print("MRAEA")

elif model == "TransEdge":
    ent_emb = tf.cast(np.load("Embeddings/TransEdge/%sent_embeds.npy" % dataset), "float32")
    print("TransEdge")

elif model == "RREA":
    ent_emb = tf.cast(np.load("Embeddings/RREA/%sent_emb.npy" % dataset), "float32")
    print("RREA")

else:
    print("The model is missed.")
    exit()

# decoding algorithm
# Triple Feature Propagation based on the entity embedding
node_size, rel_size, ent_tuple, triples_idx, ent_ent, ent_ent_val, rel_ent, ent_rel = load_graph(path)

print("Begin to Triple Feature Propagate:")
Triple_FP = TripleFeaturePropagation(train_pair, ent_emb)
start_time = time.time()
features = Triple_FP.propagation(node_size, rel_size, ent_tuple, triples_idx, ent_ent, ent_ent_val, rel_ent, ent_rel)
print(time.time()-start_time)

sims = cal_sims(test_pair,features)
sims = tf.exp(sims/0.02)

for k in range(15):
    sims = sims / tf.reduce_sum(sims,axis=1,keepdims=True)
    sims = sims / tf.reduce_sum(sims,axis=0,keepdims=True)
test(sims,"sinkhorn")

# the results of base model
csls_sims(test_pair,ent_emb)
