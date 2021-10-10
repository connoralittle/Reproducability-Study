from tensorflow.keras.layers import Dropout
from tensorflow.keras import Model
import tensorflow as tf
from spektral.layers import GATConv, GCNConv, GraphSageConv

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
        #0*self.lam is just there so lam is technically included in the first iteration so the warnings shut up
        return 0*self.lam + tf.foldl(lambda prev_adj, next_adj: (1-self.lam)*prev_adj+self.lam*next_adj, adj)