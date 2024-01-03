# _*_ coding:utf-8 _*_
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from utils import *
import tensorly
import json
import os
import faiss

import scipy.sparse as sp

seed = 10086
np.random.seed(seed)

# choose the GPU, "-1" represents using the CPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
tensorly.set_backend('tensorflow')

gpus = tf.config.experimental.list_physical_devices(device_type="GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# choose the base model and dataset
model = ["Dual_AMN", "TransEdge", "RSN"][2]
dataset = ["DBP_ZH_EN/", "DBP_JA_EN/", "DBP_FR_EN/", "SRPRS_FR_EN/", "SRPRS_DE_EN/"][3]

if "DBP" in dataset:
    path = "./EA_datasets/" + ("sharing/" if model == "TransEdge" else "mapping/") + dataset + "0_3/"
else:
    path = "./EA_datasets/" + ("sharing/" if model == "TransEdge" else "mapping/") + dataset

train_pair, test_pair = load_aligned_pair(path, ratio=0.3)

if model != "TransEdge":
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
    else:
        ent_emb = tf.cast(np.load("Embeddings/Dual_AMN/%sent_emb.npy" % dataset), "float32")
        print("Dual_AMN")
else:
    ent_emb = tf.cast(np.load("Embeddings/TransEdge/%sent_embeds.npy" % dataset), "float32")
    print("TransEdge")

# decoding algorithm
ent_dim, depth, top_k = 1024, 2, 500
if "EN" in dataset:
    rel_dim, mini_dim = ent_dim // 2, 16
else:
    rel_dim, mini_dim = ent_dim // 3, 16

node_size, rel_size, ent_tuple, triples_idx, ent_ent, ent_ent_val, rel_ent, ent_rel = load_graph(path)

def get_features(train_pair, initial_feature):

    ent_feature = initial_feature

    rel_feature = tf.zeros((rel_size, ent_feature.shape[-1]))

    ent_ent_graph = sp.coo_matrix((ent_ent_val, ent_ent.transpose()), shape=(node_size, node_size))
    ent_ent_graph = normalize_adj(ent_ent_graph)
    ent_ent_graph = convert_sparse_matrix_to_sparse_tensor(ent_ent_graph)

    rel_ent_graph = sp.coo_matrix((K.ones(rel_ent.shape[0]), rel_ent.transpose()), shape=(rel_size, node_size))
    rel_ent_graph = normalize_adj(rel_ent_graph)
    rel_ent_graph = convert_sparse_matrix_to_sparse_tensor(rel_ent_graph)

    ent_rel_graph = sp.coo_matrix((K.ones(ent_rel.shape[0]), ent_rel.transpose()), shape=(node_size, rel_size))
    ent_rel_graph = normalize_adj(ent_rel_graph)
    ent_rel_graph = convert_sparse_matrix_to_sparse_tensor(ent_rel_graph)

    ent_list, rel_list = [ent_feature], [rel_feature]
    for i in range(1): # Dual-AMN iteration: 11:81.59, 12:81.62, 13:81.6, .
        new_rel_feature = batch_sparse_matmul(rel_ent_graph, ent_feature)
        new_rel_feature = tf.nn.l2_normalize(new_rel_feature, axis=-1)

        new_ent_feature = batch_sparse_matmul(ent_ent_graph, ent_feature)
        new_ent_feature = new_ent_feature.numpy()
        ori_feature = initial_feature.numpy()
        new_ent_feature[train_pair[:, 0]] = ori_feature[train_pair[:, 0]]
        new_ent_feature[train_pair[:, 1]] = ori_feature[train_pair[:, 1]]
        new_ent_feature += batch_sparse_matmul(ent_rel_graph, rel_feature)

        new_ent_feature = tf.nn.l2_normalize(new_ent_feature, axis=-1)

        ent_feature = new_ent_feature
        rel_feature = new_rel_feature
        ent_list.append(ent_feature)
        rel_list.append(rel_feature)

    ent_feature = K.l2_normalize(K.concatenate(ent_list, 1), -1)
    rel_feature = K.l2_normalize(K.concatenate(rel_list, 1), -1)
    rel_feature = random_projection(rel_feature, rel_dim)

    batch_size = ent_feature.shape[-1] // mini_dim
    sparse_graph = tf.SparseTensor(indices=triples_idx, values=K.ones(triples_idx.shape[0]),
                                   dense_shape=(np.max(triples_idx) + 1, rel_size))
    adj_value = batch_sparse_matmul(sparse_graph, rel_feature)

    features_list = []
    for batch in range(rel_dim // batch_size + 1):
        temp_list = []
        for head in range(batch_size):
            if batch * batch_size + head >= rel_dim:
                break
            sparse_graph = tf.SparseTensor(indices=ent_tuple, values=adj_value[:, batch * batch_size + head],
                                           dense_shape=(node_size, node_size))
            feature = batch_sparse_matmul(sparse_graph, random_projection(ent_feature, mini_dim))
            temp_list.append(feature)
        if len(temp_list):
            features_list.append(K.concatenate(temp_list, -1).numpy())
    features = np.concatenate(features_list, axis=-1)

    faiss.normalize_L2(features)
    # features = ent_feature
    features = np.concatenate([ent_feature, features], axis=-1)
    return features

print("Begin to Triple Feature Propagate:")
features = get_features(train_pair, ent_emb)

sims = cal_sims(test_pair,features)
sims = tf.exp(sims/0.02)

for k in range(15):
    sims = sims / tf.reduce_sum(sims,axis=1,keepdims=True)
    sims = sims / tf.reduce_sum(sims,axis=0,keepdims=True)
test(sims,"sinkhorn")

# the results of base model
csls_sims(test_pair,ent_emb)
