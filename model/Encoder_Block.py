import math

import tensorflow as tf





def encoder_block(inputs, scope, reuse,regularizer, num_conv_layers, kernel_size, number_blocks,dropout,input_projection=False):
    with tf.variable_scope(scope,reuse=reuse):
        if input_projection:
            print("input shape measured in encoder block:",inputs.shape.as_list())
            print("Input project have not been developed yet")
        outputs=inputs
        for i in range(number_blocks): #总共需要多少个Encoder_block
            # (1)Posision Encoding
            outputs=position_encoding(outputs)
            print("signal added outputs:",outputs)

            # (2)Conv Block
            with tf.variable_scope("encoder_block_%d"%i,reuse=reuse):
                outputs=tf.expand_dims(outputs,axis=2) #(b,q,1,emb)
                print(outputs)
                l, L =(1,1)
                for j in range(num_conv_layers): # repeat
                    residual = outputs
                    #(2.1)Layernorm
                    out = norm_fn(x=outputs, scope="layer_norm_%d" %j, reuse=reuse,regularizer=regularizer) #FIXME: out should be output
                    outputs = tf.nn.dropout(outputs, 1.0 - dropout)
                    #(2.2)Multihead attention
                    outputs=multihead_attention(reuse=reuse)


def multihead_attention(reuse,scope="Multi_Head_Attention"):
    # TODO：源自attention is all you need 3.2.2 Multihead Attention
    # with tf.variable_scope()
    pass

def norm_fn(x, scope, reuse,regularizer, filters=None,epsilon=12-6):
    # 将输入的向量泛化
    # x：(b,q,1,emb)
    if filters is None:
        filters=x.get_shape()[-1]
    with tf.variable_scope(scope,default_name="layer_norm", values=[x], reuse=reuse):
        scale= tf.get_variable(
            "layer_norm_scale", [filters], regularizer = regularizer, initializer=tf.ones_initializer()) # （500,）
        bias = tf.get_variable(
            "layer_norm_bias", [filters], regularizer=regularizer, initializer=tf.zeros_initializer())# （500,）
        print("scale/bias from norm_%d:"%int(scope[-1]),scale)
        # 均值
        mean=tf.reduce_mean(x,axis=[-1],keepdims=True) # (b,q,1,emb) => (b,q,1,1) | if not keepdims => (b,q,1)
        # 均方差
        variance = tf.reduce_mean(tf.square(x - mean), axis=[-1], keepdims=True) # (b,q,1,1)
        # 泛化操作
        norm_x = (x - mean) * tf.rsqrt(variance + epsilon) #(b,q,1,emb)
        return norm_x * scale + bias
print("test")

def position_encoding(outputs):
    # TODO: 源自 attention is all you need 3.5 position encoding
    outputs_shape = tf.shape(outputs)  # tf.shape() return a tensor outputs.shape returns a tuple
    print("output shape:", outputs.shape)  # (b,q,emb)
    print("outputs shape tensor:", outputs_shape)  # (3,)

    length = outputs_shape[1]  # q for ques_emb
    channels = outputs_shape[2]  # emb for ques_emb
    # get position encoding
    position = tf.to_float(tf.range(length))  # [0,1,2....q-1]
    num_timescales = channels / 2  # emb /2
    log_timescale_increment = (
            math.log(float(1.0e4)) / (tf.to_float(num_timescales) - 1)  # math.log()以e为底的对数 FIXME: why -1
    )  # log_e(10000) / (emb/2)
    inv_timescales = tf.exp(
        tf.to_float(tf.range(num_timescales)) * -log_timescale_increment
    )  # e^ -[list([1,2..emb/2-1])*(log_e(10000) / (emb/2))] = 10000^-[list([1,2...emb/2-1])*2/emb] = 1/(10000^[list([1,2...emb/2-1])*2/emb])
    scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(inv_timescales,
                                                               0)  # pos/(10000^[list([1,2...emb/2-1])*2/emb]) => (q,1)*(1,emb/2)=>(q,emb/2)
    signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)  # (q,emb/2),(q,emb/2) =concat=> (q,emb)
    signal = tf.pad(signal, [[0, 0], [0, tf.mod(channels, 2)]])
    signal = tf.reshape(signal, [1, length, channels])  # (1,q,emb)
    print("signal:", signal)

    outputs = outputs + signal  # 自动广播？ | signal shape[1] shape[2] 与 outputs相等 | (b,q,emb)
    return outputs


