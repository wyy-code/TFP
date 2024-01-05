import numba as nb
import numpy as np
import faiss
import tensorflow as tf
import tensorflow.keras.backend as K
import pickle
import os
from evaluate import evaluate

import scipy.sparse as sp
from scipy.sparse import lil_matrix

def load_triples(file_path,reverse = True):
    @nb.njit
    def reverse_triples(triples):
        reversed_triples = np.zeros_like(triples)
        for i in range(len(triples)):
            reversed_triples[i,0] = triples[i,2]
            reversed_triples[i,2] = triples[i,0]
            if reverse:
                reversed_triples[i,1] = triples[i,1] + rel_size
            else:
                reversed_triples[i,1] = triples[i,1]
        return reversed_triples
    
    with open(file_path + "triples_1") as f:
        triples1 = f.readlines()
        
    with open(file_path + "triples_2") as f:
        triples2 = f.readlines()
        
    triples = np.array([line.replace("\n","").split("\t") for line in triples1 + triples2]).astype(np.int64)
    node_size = max([np.max(triples[:,0]),np.max(triples[:,2])]) + 1
    rel_size = np.max(triples[:,1]) + 1
    
    all_triples = np.concatenate([triples,reverse_triples(triples)],axis=0)
    all_triples = np.unique(all_triples,axis=0)
    
    return all_triples, node_size, rel_size*2 if reverse else rel_size

def load_aligned_pair(file_path):
    with open(file_path + "ref_ent_ids") as f:
        ref = f.readlines()
    with open(file_path + "sup_ent_ids") as f:
        sup = f.readlines()

        
    ref = np.array([line.replace("\n","").split("\t") for line in ref]).astype(np.int64)
    sup = np.array([line.replace("\n","").split("\t") for line in sup]).astype(np.int64)

    return sup, ref

def test(sims,mode = "sinkhorn",batch_size = 1024):
    
    if mode == "sinkhorn":
        results = []
        for epoch in range(len(sims) // batch_size + 1):
            sim = sims[epoch*batch_size:(epoch+1)*batch_size]
            rank = tf.argsort(-sim,axis=-1)
            ans_rank = np.array([i for i in range(epoch * batch_size,min((epoch+1) * batch_size,len(sims)))])
            results.append(tf.where(tf.equal(tf.cast(rank,ans_rank.dtype),tf.tile(np.expand_dims(ans_rank,axis=1),[1,len(sims)]))).numpy())
        results = np.concatenate(results,axis=0)
        
        @nb.jit(nopython = True)
        def cal(results):
            hits1,hits10,mrr = 0,0,0
            for x in results[:,1]:
                if x < 1:
                    hits1 += 1
                if x < 10:
                    hits10 += 1
                mrr += 1/(x + 1)
            return hits1,hits10,mrr
        hits1,hits10,mrr = cal(results)
        print("hits@1 : %.2f%% hits@10 : %.2f%% MRR : %.2f%%" % (hits1/len(sims)*100,hits10/len(sims)*100,mrr/len(sims)*100))
        #return [hits1/len(sims)*100,hits10/len(sims)*100,mrr/len(sims)*100]
    else:
        c = 0
        for i,j in enumerate(sims[1]):
            if i == j:
                c += 1
        print("hits@1 : %.2f%%"%(100 * c/len(sims[0])))

def normalize_adj(adj):
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -1).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return d_mat_inv_sqrt.dot(adj)

def batch_sparse_matmul(sparse_tensor, dense_tensor, batch_size=128, save_mem=False):
    results = []
    for i in range(dense_tensor.shape[-1] // batch_size + 1):
        temp_result = tf.sparse.sparse_dense_matmul(sparse_tensor, dense_tensor[:, i * batch_size:(i + 1) * batch_size])
        if save_mem:
            temp_result = temp_result.numpy()
        results.append(temp_result)
    if save_mem:
        return np.concatenate(results, -1)
    else:
        return K.concatenate(results, -1)
def convert_sparse_matrix_to_sparse_tensor(X):
    coo = X.tocoo().astype(np.float32)
    indices = np.mat([coo.row, coo.col]).transpose()
    return tf.SparseTensor(indices, coo.data, coo.shape)
def load_graph(path):
    if os.path.exists(path + "graph_cache.pkl"):
        return pickle.load(open(path + "graph_cache.pkl", "rb"))

    triples = []
    with open(path + "triples_1") as f:
        for line in f.readlines():
            h, r, t = [int(x) for x in line.strip().split("\t")]
            triples.append([h, t, 2 * r])
            triples.append([t, h, 2 * r + 1])
    with open(path + "triples_2") as f:
        for line in f.readlines():
            h, r, t = [int(x) for x in line.strip().split("\t")]
            triples.append([h, t, 2 * r])
            triples.append([t, h, 2 * r + 1])
    triples = np.unique(triples, axis=0)
    node_size, rel_size = np.max(triples) + 1, np.max(triples[:, 2]) + 1

    ent_tuple, triples_idx = [], []
    ent_ent_s, rel_ent_s, ent_rel_s = {}, set(), set()
    last, index = (-1, -1), -1

    adj_matrix = lil_matrix((node_size, node_size))
    for h, r, t in triples:
        adj_matrix[h, t] = 1
        adj_matrix[t, h] = 1

    for i in range(node_size):
        ent_ent_s[(i, i)] = 0

    for h, t, r in triples:
        ent_ent_s[(h, h)] += 1
        ent_ent_s[(t, t)] += 1

        if (h, t) != last:
            last = (h, t)
            index += 1
            ent_tuple.append([h, t])
            ent_ent_s[(h, t)] = 0

        triples_idx.append([index, r])
        ent_ent_s[(h, t)] += 1
        rel_ent_s.add((r, h))
        ent_rel_s.add((t, r))

    ent_tuple = np.array(ent_tuple)
    triples_idx = np.unique(np.array(triples_idx), axis=0)


    ent_ent = np.unique(np.array(list(ent_ent_s.keys())), axis=0)
    ent_ent_val = np.array([ent_ent_s[(x, y)] for x, y in ent_ent]).astype("float32")
    rel_ent = np.unique(np.array(list(rel_ent_s)), axis=0)
    ent_rel = np.unique(np.array(list(ent_rel_s)), axis=0)

    # graph_data = [node_size, rel_size, ent_tuple, triples_idx, ent_ent, ent_ent_val, rel_ent, ent_rel]
    graph_data = [node_size, rel_size, ent_tuple, triples_idx, ent_ent, ent_ent_val, rel_ent, ent_rel]
    pickle.dump(graph_data, open(path + "graph_cache.pkl", "wb"))
    return graph_data

def sparse_sinkhorn_sims(left, right, features, top_k=500, iteration=15, mode="test"):
    features_l = features[left]
    features_r = features[right]

    faiss.normalize_L2(features_l);
    faiss.normalize_L2(features_r)

    dim, measure = features_l.shape[1], faiss.METRIC_INNER_PRODUCT
    if mode == "test":
        param = 'Flat'
        index = faiss.index_factory(dim, param, measure)
    else:
        param = 'IVF256(RCQ2x5),PQ32'
        index = faiss.index_factory(dim, param, measure)
        index.nprobe = 16
    # if len(gpus):
    #     index = faiss.index_cpu_to_gpu(res, 0, index)
    index.train(features_r)
    index.add(features_r)
    sims, index = index.search(features_l, top_k)

    row_sims = K.exp(sims.flatten() / 0.02)
    index = K.flatten(index.astype("int32"))

    size = len(left)
    row_index = K.transpose(([K.arange(size * top_k) // top_k, index, K.arange(size * top_k)]))
    col_index = tf.gather(row_index, tf.argsort(row_index[:, 1]))
    covert_idx = tf.argsort(col_index[:, 2])

    for _ in range(iteration):
        row_sims = row_sims / tf.gather(indices=row_index[:, 0], params=tf.math.segment_sum(row_sims, row_index[:, 0]))
        col_sims = tf.gather(row_sims, col_index[:, 2])
        col_sims = col_sims / tf.gather(indices=col_index[:, 1], params=tf.math.segment_sum(col_sims, col_index[:, 1]))
        row_sims = tf.gather(col_sims, covert_idx)

    return K.reshape(row_index[:, 1], (-1, top_k)), K.reshape(row_sims, (-1, top_k))

def random_projection(x,out_dim):
    random_vec = K.l2_normalize(tf.random.normal((x.shape[-1],out_dim),mean=0,stddev=(1/out_dim)**0.5),axis=-1)
    return K.dot(x,random_vec)

def cal_sims(test_pair,feature):
    feature = tf.nn.l2_normalize(feature,axis = -1)
    feature_a = tf.gather(indices=test_pair[:,0],params=feature)
    feature_b = tf.gather(indices=test_pair[:,1],params=feature)
    return tf.matmul(feature_a,tf.transpose(feature_b,[1,0]))

def csls_sims(test_pair,feature):
    evaluater = evaluate(test_pair)
    feature = tf.nn.l2_normalize(feature,axis = -1)
    feature_a = tf.gather(indices=test_pair[:,0],params=feature)
    feature_b = tf.gather(indices=test_pair[:,1],params=feature)
    evaluater.test(feature_a,feature_b)
