from models.conv_encoder_utils import *
import tensorflow as tf
from collections import namedtuple


def _create_position_embedding(position_embedding, lengths, maxlen):
    # Slice to size of current sequence
    pe_slice = position_embedding
    # Replicate encodings for each element in the batch
    batch_size = tf.shape(lengths)[0]
    pe_batch = tf.tile([pe_slice], [batch_size, 1, 1])

    # Mask out positions that are padded
    positions_mask = tf.sequence_mask(lengths=lengths, maxlen=maxlen, dtype=tf.float32)
    positions_embed = pe_batch * tf.expand_dims(positions_mask, 2)

    positions_embed = tf.reverse_sequence(positions_embed, lengths, batch_dim=0, seq_dim=1)
    # [[1,2,3,4,PAD,PAD,PAD],[2,3,PAD,PAD,PAD,PAD,PAD]]   [4,2]
    positions_embed = tf.reverse(positions_embed, [1])
    # --> [[4,3,2,1,PAD,PAD,PAD],[3,2,PAD,PAD,PAD,PAD,PAD]] --> [[PAD,PAD,PAD,1,2,3,4],[PAD,PAD,PAD,PAD,PAD,2,3]]

    return positions_embed


def conv_encode(inputs, sequence_length, position_embedding, flag_is_training, is_pos_embed):

    """ 移植之后自定义的参数 """
    position_embeddings_enable = is_pos_embed
    embedding_dropout_keep_prob = 0.9
    nhid_dropout_keep_prob = 0.9
    is_training = flag_is_training

    cnn_layers = 5

    cnn_nhids = ""
    cnn_nhid_default = 300

    cnn_kwidths = ""
    cnn_kwidth_default = 3


    #TODO """ inputs的格式还需要测试一下 """
    embed_size = inputs.get_shape().as_list()[-1]

    if position_embeddings_enable:
        positions_embed = _create_position_embedding(position_embedding,
                                                     lengths=sequence_length,  # tensor, data lengths
                                                     maxlen=tf.shape(inputs)[1])  # max len in this batch
        """_combiner_fn = tf.add() 位置向量与词向量相加"""
        inputs = tf.add(inputs, positions_embed)

    # Apply dropout to embeddings
    inputs = tf.contrib.layers.dropout(inputs=inputs, keep_prob=embedding_dropout_keep_prob, is_training=is_training)

    with tf.variable_scope("encoder_cnn"):
        next_layer = inputs
        if cnn_layers > 0:
            """ generate parameter list for convolution """
            nhids_list = parse_list_or_default(cnn_nhids, cnn_layers, cnn_nhid_default)
            kwidths_list = parse_list_or_default(cnn_kwidths, cnn_layers, cnn_kwidth_default)

            """ mapping embedding dim to hidden dim """
            next_layer = linear_mapping_weightnorm(next_layer,
                                                   nhids_list[0],
                                                   dropout=embedding_dropout_keep_prob,
                                                   var_scope_name="linear_mapping_before_cnn")

            next_layer = conv_encoder_stack(next_layer, nhids_list, kwidths_list,
                                            {'src': embedding_dropout_keep_prob, 'hid': nhid_dropout_keep_prob},
                                            mode=is_training)

            next_layer = linear_mapping_weightnorm(next_layer,
                                                   embed_size,
                                                   var_scope_name="linear_mapping_after_cnn")

        ## The encoder stack will receive gradients *twice* for each attention pass: dot product and weighted sum.
        ##cnn = nn.GradMultiply(cnn, 1 / (2 * nattn))
        cnn_c_output = (next_layer + inputs) * tf.sqrt(0.5)

    final_state = tf.reduce_mean(cnn_c_output, 1)

    EncoderOutput = namedtuple(
        "EncoderOutput",
        "outputs final_state attention_values attention_values_length")

    return EncoderOutput(
        outputs=next_layer,
        final_state=final_state,
        attention_values=cnn_c_output,
        attention_values_length=sequence_length)
