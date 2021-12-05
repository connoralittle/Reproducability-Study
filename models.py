from tensorflow.keras.layers import Dropout, Dense
from tensorflow.keras import Model
import tensorflow as tf
from spektral.layers import GATConv, GCNConv, GraphSageConv
import numpy as np

#This file contains all of the models created in tensorflow
class GAT(Model):
    def __init__(self, nhid, nclass, dropout):
        super(GAT, self).__init__()

        self.dropout = Dropout(dropout)
        self.gat1 = GATConv(channels=nhid,attn_heads=1,concat_heads=True, activation='relu')
        self.gat2 = GATConv(channels=nclass,attn_heads=1,concat_heads=True, activation='softmax')

    def call(self, inputs, training=None):
        feats, adj = inputs
        x_1 = self.gat1([feats, adj])
        dropout = self.dropout(x_1, training=training)
        return self.gat2([dropout, adj])

class GCN(Model):
    def __init__(self, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.dropout = Dropout(dropout)
        self.gcn1 = GCNConv(channels=nhid, activation='relu')
        self.gcn2 = GCNConv(channels=nclass, activation='softmax')

    def call(self, inputs, training=None):
        feats, adj = inputs
        x_1 = self.gcn1([feats, adj])
        dropout = self.dropout(x_1, training=training)
        return self.gcn2([dropout, adj])

class GraphSage(Model):
    def __init__(self, nhid, nclass, dropout):
        super(GraphSage, self).__init__()

        self.dropout = Dropout(dropout)
        self.gs1 = GraphSageConv(channels=nhid, activation='relu', aggregate_op='mean')
        self.gs2 = GraphSageConv(channels=nclass, activation='softmax', aggregate_op='mean')

    def call(self, inputs, training=None):
        feats, adj = inputs
        x_1 = self.gs1([feats, tf.sparse.from_dense(adj)])
        dropout = self.dropout(x_1, training=training)
        return self.gs2([dropout, tf.sparse.from_dense(adj)])

class RNNGCN(Model):
    def __init__(self, nhid, nclass, dropout):
        super(RNNGCN, self).__init__()

        self.updated_adj = RNNGNNLayer()
        self.dropout = Dropout(dropout)
        self.gcn1 = GCNConv(channels=nhid, activation='relu')
        self.gcn2 = GCNConv(channels=nclass, activation='softmax')

    def call(self, inputs, training=None):
        feats, adj = inputs

        adj = self.updated_adj(adj)
        x_1 = self.gcn1([feats[:,-1,:], adj])
        dropout = self.dropout(x_1, training=training)
        return self.gcn2([dropout, adj])

class RNNGNNLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(RNNGNNLayer, self).__init__()

        self.lam = tf.Variable(initial_value=0.2, trainable=True, name="lam", dtype=tf.float32)

    def call(self, adj):
        self.lam.assign(tf.clip_by_value(self.lam, 0, 1))
        return tf.foldl(lambda prev_adj, next_adj: (1-self.lam)*prev_adj+self.lam*next_adj, adj)

class TRNNGCN(Model):
    def __init__(self, nnode, nhid, nclass, dropout):
        super(TRNNGCN, self).__init__()

        self.updated_adj = TRNNGNNLayer(nclass, nnode)
        self.dropout = Dropout(dropout)
        self.gcn1 = GCNConv(channels=nhid, activation='relu')
        self.gcn2 = GCNConv(channels=nclass, activation='softmax')

    def call(self, inputs, training=None):
        feats, adj = inputs

        adj = self.updated_adj(adj)
        x_1 = self.gcn1([feats[:,-1,:], adj])
        dropout = self.dropout(x_1, training=training)
        output = self.gcn2([dropout, adj])

        amax = tf.math.argmax(output, axis=1)

        self.updated_adj.h.assign(tf.zeros(self.updated_adj.h.shape))
        self.updated_adj.h.scatter_nd_add(tf.stack([range(feats.shape[0]), amax], axis=1), tf.fill([feats.shape[0]], 1.0))

        return output

class TRNNGNNLayer(tf.keras.layers.Layer):
    def __init__(self, nclass, nnode):
        super(TRNNGNNLayer, self).__init__()

        self.lam = tf.Variable(tf.fill([nclass, nclass], 0.5), trainable=True, name="lam", dtype=tf.float32)

        self.h = tf.Variable(tf.zeros([nnode, nclass]), trainable=False, dtype=tf.float32)
        self.h.scatter_nd_add(tf.stack([range(nnode), tf.random.uniform([nnode], 0, nclass, dtype=tf.int32, seed=5)], axis=1), tf.fill([nnode], 1.0))


    def call(self, adj):
        #Set boundary conditions
        self.lam.assign(tf.clip_by_value(self.lam, 0, 1))
        lam_temp = tf.matmul(tf.matmul(self.h, self.lam), tf.transpose(self.h))

        adj = tf.foldl(lambda prev_adj, next_adj: (1-lam_temp)*prev_adj+lam_temp*next_adj, adj)

        return adj

class GCNLSTM(Model):
    def __init__(self, nhid, nclass, dropout):
        super(GCNLSTM, self).__init__()

        self.dropout = Dropout(dropout)
        self.gcn1 = GCNConv(channels=nhid, activation='relu')
        self.gcn2 = GCNConv(channels=nclass)
        self.lstm = tf.keras.layers.LSTM(units=nclass, dropout=0.5)

    def call(self, inputs, training=None):
        feats, adj = inputs

        out = []

        for i in range(adj.shape[0]):
            x_1 = self.gcn1([feats[:,-1,:], adj[i,:,:]])
            dropout = self.dropout(x_1, training=training)
            out.append(self.gcn2([dropout, adj[i,:,:]]))
        out = tf.stack(out, 1)
        return tf.keras.activations.softmax(self.lstm(out))

class EGCN(Model): #egcn_o
    def __init__(self, nfeat, nhid, nclass, skipfeats=False):
        super().__init__()

        self.skipfeats = skipfeats
        self.GRU_layers = []
        self.mlp = tf.keras.Sequential(Dense(units = nhid, activation="relu"),
                                       Dense(units = nclass))
        
        self.GRU_layers.append(tf.keras.layers.GRU(units=nhid, activation='relu'))
        self.GRU_layers.append(tf.keras.layers.GRU(units=nhid, activation='relu'))

    def call(self,inputs):
        feats, adj = inputs

        node_feats= feats[:,-1,:]
        for unit in self.GRU_layers:
            feats = unit(adj,feats)

        out = Nodes_list[:,-1,:]
        if self.skipfeats:
            out = tf.concat((out,node_feats), dim=1)  
       
        
        return tf.nn.softmax(self.mlp(out), dim=1)