import json
import math

import tensorflow as tf

from model.Encoder_Block import encoder_block


class QA_Net():
    def __init__(self):
        self.opts=json.load(open("model/config.json"))

        self.regularizer = tf.contrib.layers.l2_regularizer(scale = 3e-7)

        self.initializer = lambda: tf.contrib.layers.variance_scaling_initializer(factor=1.0,
                                                                             mode='FAN_AVG',
                                                                             uniform=True,
                                                                             dtype=tf.float32)
        self.initializer_relu = lambda: tf.contrib.layers.variance_scaling_initializer(factor=2.0,
                                                                                  mode='FAN_IN',
                                                                                  uniform=False,
                                                                                  dtype=tf.float32)

    def random_weight(self, dim_in, dim_out, name=None, stddev=1.0):
        return tf.Variable(tf.truncated_normal([dim_in, dim_out], stddev=stddev / math.sqrt(float(dim_in))), name=name)

    def build(self):
        opt=self.opts

        # Input FIXME: opt["batch"] should be changed to None
        para=tf.placeholder(dtype=tf.int32,shape=[opt["batch"],opt["p_length"]],name="para") #(b,p)
        ques=tf.placeholder(dtype=tf.int32,shape=[opt["batch"],opt["q_length"]],name="ques") #(b,q)
        para_char=tf.placeholder(dtype=tf.int32,shape=[opt["batch"],opt["p_length"],opt["char_limit"]],name="para_char") #(b,p,c) 'c' 代表最大单词长度
        ques_char=tf.placeholder(dtype=tf.int32,shape=[opt["batch"],opt["q_length"],opt["char_limit"]],name="para_char") #(b,q,c)

        word_emb_mat=tf.placeholder(dtype=tf.float32,shape=[opt["vocab_size"],opt["emb_dim"]],name="word_embedding_matrix") # 直接读入GLoVe
        char_emb_mat=self.random_weight(dim_in=opt["char_vocab_size"],dim_out=opt["char_dim"],name="char_embedding_matrix") # 可训练的权重矩阵

        # Output
        ans_1=tf.placeholder(dtype=tf.int32,shape=[opt["batch"],opt["p_length"]],name="answer_1") #(b,p)
        ans_2=tf.placeholder(dtype=tf.int32,shape=[opt["batch"],opt["p_length"]],name="answer_2") #(b,p)


        print("Layer1: Input Embedding Layer")
        with tf.variable_scope("Input_Embedding_Layer"):

            # # Character Level Embedding
            ques_char_emb=tf.reshape(tf.nn.embedding_lookup(char_emb_mat,ques_char),shape=[opt["batch"]*opt["q_length"],opt["char_limit"],opt["char_dim"]]) # (b*q,cl,c_dim)
            para_char_emb=tf.reshape(tf.nn.embedding_lookup(char_emb_mat,para_char),shape=[opt["batch"]*opt["p_length"],opt["char_limit"],opt["char_dim"]]) # (b*p,cl,c_dim)

            ques_char_emb=tf.nn.dropout(ques_char_emb,1.0 - 0.5 * opt["dropout"])
            para_char_emb=tf.nn.dropout(para_char_emb,1.0 - 0.5 * opt["dropout"])
            print("para char emb RESHAPED:",para_char_emb) # (b*p,cl,c_dim)
            print("ques char emb RESHAPED:",ques_char_emb) # (b*q,cl,c_dim)

            # conv highway encoder
            filter_shape=[5,opt["char_dim"],opt["char_emb_size"]] # (k,c_dim,c_emb)
            bias_shape=[1,1,opt["char_emb_size"]] #(1,1,c_emb)
            with tf.variable_scope("char_conv"):
                kernel_=tf.get_variable(name="kernel_",
                                        shape=filter_shape,
                                        dtype=tf.float32,
                                        regularizer=self.regularizer,
                                        initializer=self.initializer_relu())
                bias_=tf.get_variable(name="bias_",
                                      shape=bias_shape,
                                      regularizer=self.regularizer,
                                      initializer=tf.zeros_initializer())
                print("kernel:",kernel_) # (5,c_dim,c_emb)
                print("CONV: (16,64) =conv1d=kernel(5,64,200)=> (16-5+1,200)")
                para_char_emb=tf.nn.relu(tf.nn.conv1d(value=para_char_emb,filters=kernel_,stride=1,padding="VALID")+bias_)
                ques_char_emb=tf.nn.relu(tf.nn.conv1d(value=ques_char_emb,filters=kernel_,stride=1,padding="VALID")+bias_)

            print("conv para emb:",para_char_emb) # (b*p,c_dim_cov,c_emb) | padding=VALID: c_dim_conv = ceil( cl - k + 1 )/stride | c_dim_cov 表示c_dim 被 k 卷积之后的长度
            print("conv ques emb:",ques_char_emb) # (b*q,c_dim_cov,c_emb)

            # take the maximum value of each row
            para_char_emb=tf.reduce_max(para_char_emb,axis=1) # (b*p,c_dim_cov,c_emb) => (b*q,c_emb)
            ques_char_emb=tf.reduce_max(ques_char_emb,axis=1) # (b*q,c_dim_cov,c_emb) => (b*q,c_emb)
            print("max conv para emb;",para_char_emb)
            print("max conv ques emb;",ques_char_emb)

            para_char_emb=tf.reshape(para_char_emb,shape=[opt["batch"],opt["p_length"],-1]) # (b*p,c_emb)=>(b,p,c_emb)
            ques_char_emb=tf.reshape(ques_char_emb,shape=[opt["batch"],opt["q_length"],-1]) #(b*q,c_emb)=>(b,q,c_emb)
            print("para char emb RECOVERED:",para_char_emb)
            print("ques char emb RECOVERED:",ques_char_emb)

            # # Word Level Embedding
            ques_word_emb=tf.nn.embedding_lookup(word_emb_mat,ques) # (b,q,w_emb)
            para_word_emb=tf.nn.embedding_lookup(word_emb_mat,para) # (b,p,w_emb)
            ques_word_emb=tf.nn.dropout(ques_word_emb,1.0-opt["dropout"])
            para_word_emb=tf.nn.dropout(para_word_emb,1.0-opt["dropout"])
            print("para word emb:",para_word_emb)
            print("ques word emb:",ques_word_emb)

            # Word + Character Level Embedding
            para_emb=tf.concat([para_word_emb,para_char_emb],axis=2) # (b,q,w_emb+c_emb) => (b.q.emb)
            ques_emb=tf.concat([ques_word_emb,ques_char_emb],axis=2) # (b,q,w_emb+c_emb)=> (b,q,emb)
            print("para emb:",para_emb)
            print("ques emb:",ques_emb)
            print()

        print("Layer2: Embedding Encode Layer")
        with tf.variable_scope("Embedding_Encoding_Layer"):
            ques_enc=encoder_block(inputs=ques_emb,
                                   regularizer=self.regularizer,
                                   number_blocks=1,
                                   kernel_size=7,
                                   num_conv_layers=4,
                                   reuse=False,
                                   dropout=opt["dropout"],
                                   scope="Encoder_Block")



