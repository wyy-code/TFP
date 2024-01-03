# _*_ coding:utf-8 _*_
import numpy as np
from sklearn.decomposition import TruncatedSVD
from tqdm import tqdm
from scipy import optimize
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
model = ["Dual_AMN", "TransEdge", "RSN"][0]
dataset = ["DBP_ZH_EN/", "DBP_JA_EN/", "DBP_FR_EN/", "SRPRS_FR_EN/", "SRPRS_DE_EN/"][0]

if "DBP" in dataset:
    path = "./EA_datasets/" + ("sharing/" if model == "TransEdge" else "mapping/") + dataset + "0_3/"
else:
    path = "./EA_datasets/" + ("sharing/" if model == "TransEdge" else "mapping/") + dataset

train_pair, test_pair = load_aligned_pair(path, ratio=0.3)

# build the adjacency sparse tensor of KGs and load the initial embeddings
triples = []

flag = model == "Dual_AMN"
with open(path + "triples_1") as f:
    for line in f.readlines():
        h, r, t = [int(x) for x in line.strip().split("\t")]
        triples.append([h, t, r + flag])
with open(path + "triples_2") as f:
    for line in f.readlines():
        h, r, t = [int(x) for x in line.strip().split("\t")]
        triples.append([h, t, r + flag])

if model != "TransEdge":
    triples = np.array(triples)
    triples = np.unique(triples, axis=0)
    node_size, rel_size = np.max(triples[:, 0]) + 1, np.max(triples[:, 2]) + 1
    triples = np.concatenate([triples, [(t, h, r + rel_size) for h, t, r in triples]], axis=0)
    rel_size = rel_size * 2

    if model == "RSN":
        print(train_pair)
        print(test_pair)
        all_pair = np.concatenate((train_pair, test_pair))
        print(all_pair)
        emb_path = "Embeddings/RSN/%s" % dataset
        ent_emb = tf.cast(np.load(emb_path + "ent_emb.npy"), "float32")
        rel_emb = tf.cast(np.load(emb_path + "rel_emb.npy"), "float32")
        ent_dic, rel_dic = json.load(open(emb_path + "ent_id2id.json")), json.load(open(emb_path + "rel_id2id.json"))
        new_triples, new_test = [], []
        for h, t, r in triples:
            new_triples.append([int(ent_dic[str(h)]), int(ent_dic[str(t)]), int(rel_dic[str(r)])])
        for a, b in test_pair:
            new_test.append([int(ent_dic[str(a)]), int(ent_dic[str(b)])])
        triples = np.array(new_triples)
        test_pair = np.array(new_test)
    else:
        triples = np.concatenate([triples, [(t, t, 0) for t in range(node_size)]], axis=0)
        ent_emb = tf.cast(np.load("Embeddings/Dual_AMN/%sent_emb.npy" % dataset), "float32")
        rel_emb = tf.cast(np.load("Embeddings/Dual_AMN/%srel_emb.npy" % dataset), "float32")

    triples = np.unique(triples, axis=0)

else:
    triples = np.array(triples)
    triples = np.unique(triples, axis=0)
    node_size, rel_size = np.max(triples) + 1, np.max(triples[:, 2]) + 1
    triples = np.concatenate([triples, [(t, h, r) for h, t, r in triples]], axis=0)
    triples = np.unique(triples, axis=0)
    ent_emb = tf.cast(np.load("Embeddings/TransEdge/%sent_embeds.npy" % dataset), "float32")
    rel_emb = tf.cast(np.load("Embeddings/TransEdge/%srel_embeds.npy" % dataset), "float32")


# decoding algorithm
ent_dim, depth, top_k = 1024, 2, 500
if "EN" in dataset:
    rel_dim, mini_dim = ent_dim // 2, 16
else:
    rel_dim, mini_dim = ent_dim // 3, 16

node_size, rel_size, ent_tuple, triples_idx, ent_ent, ent_ent_val, rel_ent, ent_rel = load_graph(path)
candidates_x,candidates_y = set([x for x,y in test_pair]), set([y for x,y in test_pair])


def get_features(train_pair, extra_feature=None):
    if extra_feature is not None:
        ent_feature = extra_feature
    else:
        random_vec = K.l2_normalize(tf.random.normal((len(train_pair), ent_dim)), axis=-1)
        ent_feature = tf.tensor_scatter_nd_update(tf.zeros((node_size, ent_dim)), train_pair.reshape((-1, 1)),
                                                  tf.repeat(random_vec, 2, axis=0))
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

    # # ########Origin graph
    # ent_ent_graph = tf.SparseTensor(indices=ent_ent, values=ent_ent_val, dense_shape=(node_size, node_size))
    # rel_ent_graph = tf.SparseTensor(indices=rel_ent, values=K.ones(rel_ent.shape[0]), dense_shape=(rel_size, node_size))
    # ent_rel_graph = tf.SparseTensor(indices=ent_rel, values=K.ones(ent_rel.shape[0]), dense_shape=(node_size, rel_size))


    ent_list, rel_list = [ent_feature], [rel_feature]
    for i in range(1): # origin iteration is 2.
        new_rel_feature = batch_sparse_matmul(rel_ent_graph, ent_feature)
        new_rel_feature = tf.nn.l2_normalize(new_rel_feature, axis=-1)

        if extra_feature is not None:
            new_ent_feature = batch_sparse_matmul(ent_ent_graph, ent_feature)
            new_ent_feature = new_ent_feature.numpy()
            ori_feature = extra_feature.numpy()
            new_ent_feature[train_pair[:, 0]] = ori_feature[train_pair[:, 0]]
            new_ent_feature[train_pair[:, 1]] = ori_feature[train_pair[:, 1]]
            new_ent_feature += batch_sparse_matmul(ent_rel_graph, rel_feature)
        else:
            new_ent_feature = batch_sparse_matmul(ent_ent_graph, ent_feature)
            new_ent_feature += batch_sparse_matmul(ent_rel_graph, rel_feature)


        new_ent_feature = tf.nn.l2_normalize(new_ent_feature, axis=-1)

        ent_feature = new_ent_feature;
        rel_feature = new_rel_feature
        ent_list.append(ent_feature);
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
    if extra_feature is not None:
        features = np.concatenate([ent_feature, features], axis=-1)
    return features


epochs = 3
for epoch in range(epochs):
    print("Round %d start:" % (epoch + 1))
    s_features = get_features(train_pair)
    l_features = get_features(train_pair, extra_feature=ent_emb)

    features = np.concatenate([s_features, l_features], -1)

    if epoch < epochs - 1:
        left, right = list(candidates_x), list(candidates_y)
        index, sims = sparse_sinkhorn_sims(left, right, features, top_k)
        ranks = tf.argsort(-sims, -1).numpy()
        sims = sims.numpy();
        index = index.numpy()

        temp_pair = []
        x_list, y_list = list(candidates_x), list(candidates_y)
        for i in range(ranks.shape[0]):
            if sims[i, ranks[i, 0]] > 0.9:
                x = x_list[i]
                y = y_list[index[i, ranks[i, 0]]]
                temp_pair.append((x, y))

        for x, y in temp_pair:
            if x in candidates_x:
                candidates_x.remove(x);
            if y in candidates_y:
                candidates_y.remove(y);

        print("new generated pairs = %d" % (len(temp_pair)))
        print("rest pairs = %d" % (len(candidates_x)))

        if not len(temp_pair):
            break
        train_pair = np.concatenate([train_pair, np.array(temp_pair)])

    # right_list, wrong_list = test(test_pair, features, top_k)

    sims = cal_sims(test_pair,features)
    sims = tf.exp(sims/0.02)
    print("Begin to scale the sim...")
    # sk = skp.SinkhornKnopp()
    # sims = sk.fit(sims)
    for k in range(15):
        sims = sims / tf.reduce_sum(sims,axis=1,keepdims=True)
        sims = sims / tf.reduce_sum(sims,axis=0,keepdims=True)
    test(sims,"sinkhorn")

# the results of base model
csls_sims(test_pair,ent_emb)
