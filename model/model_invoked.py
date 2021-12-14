# -*- coding: utf-8 -*-
# /usr/bin/python3
'''
Feb. 2019 by kyubyong park.
kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/transformer
Transformer network
'''
import tensorflow as tf

# from new_data_loader import *
from extract_data_subword import *
from modules import get_token_embeddings, get_doc_embeddings, ff, positional_encoding, multihead_attention, label_smoothing, noam_scheme
# from utils import convert_idx_to_token_tensor
#from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)

class Transformer:
    '''
    xs: tuple of
        x: int32 tensor. (N, T1)
        x_seqlens: int32 tensor. (N,)
        sents1: str tensor. (N,)
    ys: tuple of
        decoder_input: int32 tensor. (N, T2)
        y: int32 tensor. (N, T2)
        y_seqlen: int32 tensor. (N, )
        sents2: str tensor. (N,)
    training: boolean.
    '''
    def __init__(self, hp):
        self.hp = hp 
        # self.data = data_loader(hp.data_path, hp.data_path)
        self.data = localContext(
        hp.body_context_size, 
        hp.doc_context_size, 
        hp.project_context_size,
        hp.tgt_name_size,
        # hp.vocab_file, 
        hp.sub_word_vocab_file, 
        hp.doc_vocab_file, 
        include_docstring=True, 
        expr_max_len=1024, 
        expr_max_num=30,
        datapath = hp.data_path
    )

        self.embeddings = get_token_embeddings(len(self.data.w2id), self.hp.d_model, zero_pad=True)
        self.doc_embeddings = get_doc_embeddings(len(self.data.doc_w2id), self.hp.d_model, zero_pad=True)

        # encoder part
        self.body_batch = tf.placeholder(tf.int32, [hp.batch_size, None], name='body_batch')
        self.pro_batch = tf.placeholder(tf.int32, [hp.batch_size, None], name='pro_batch')
        self.doc_batch = tf.placeholder(tf.int32, [hp.batch_size, None], name='doc_batch')
        self.invoked_batch = tf.placeholder(tf.float32, [hp.batch_size, None], name='invoked_batch')

        # self._enc_lens = tf.placeholder(tf.int32, [hp.batch_size], name='enc_lens')

        # decoder part
        self.dec_inp_batch = tf.placeholder(tf.int32, [hp.batch_size, self.data.tgt_name_len], name='dec_batch')
        self.dec_tgt_batch = tf.placeholder(tf.int32, [hp.batch_size, self.data.tgt_name_len], name='target_batch')
        # self._dec_padding_mask = tf.placeholder(tf.float32, [hp.batch_size, hp.max_dec_steps], name='dec_padding_mask')

    def encode(self, training=True):
        '''
        Returns
        memory: encoder outputs. (N, T1, d_model)
        '''
        with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE):
            body_x, pro_x, doc_x = self.body_batch, self.pro_batch, self.doc_batch
            if self.hp.pro:
                self.concat_x = tf.concat((pro_x, body_x), -1)
                # src_masks
                src_masks = tf.math.equal(self.concat_x, self.data.PAD) # (N, T1)
                pro_src_masks = tf.math.equal(pro_x, 0)

                # embedding
                enc = tf.nn.embedding_lookup(self.embeddings, self.concat_x) # (N, T1, d_model)
                enc *= self.hp.d_model**0.5 # scale
                enc += positional_encoding(enc, self.data.body_context_size+self.data.project_context_size)

                cxt_enc = tf.nn.embedding_lookup(self.embeddings, pro_x) # (N, T1, d_model)
                cxt_enc *= self.hp.d_model**0.5 # scale

                cxt_enc += positional_encoding(cxt_enc, self.data.project_context_size)
                cxt_enc = tf.layers.dropout(cxt_enc, self.hp.dropout_rate, training=training)

            else:
                src_masks = tf.math.equal(body_x, self.data.PAD) # (N, T1)
                # pro_src_masks = tf.math.equal(body_x, 0)

                # embedding
                enc = tf.nn.embedding_lookup(self.embeddings, body_x) # (N, T1, d_model)
                enc *= self.hp.d_model**0.5 # scale
                enc += positional_encoding(enc, self.data.body_context_size)

            doc_src_masks = tf.math.equal(doc_x, 0) # (N, T1)
            doc_enc = tf.nn.embedding_lookup(self.doc_embeddings, doc_x) # (N, T1, d_model)
            doc_enc *= self.hp.d_model**0.5 # scale

            doc_enc += positional_encoding(doc_enc, self.data.doc_context_size)
            doc_enc = tf.layers.dropout(doc_enc, self.hp.dropout_rate, training=training)

            enc = tf.concat((enc, doc_enc), 1)
            src_masks = tf.concat((src_masks, doc_src_masks), -1)

            # cxt_enc = tf.nn.embedding_lookup(self.embeddings, pro_x) # (N, T1, d_model)
            # cxt_enc *= self.hp.d_model**0.5 # scale

            # cxt_enc += positional_encoding(cxt_enc, self.data.project_context_size)
            # cxt_enc = tf.layers.dropout(cxt_enc, self.hp.dropout_rate, training=training)


            enc = tf.layers.dropout(enc, self.hp.dropout_rate, training=training)

            ## Blocks
            for i in range(self.hp.num_blocks):
                with tf.variable_scope("num_blocks_{}".format(i), reuse=tf.AUTO_REUSE):
                    # self-attention
                    enc = multihead_attention(queries=enc,
                                              keys=enc,
                                              values=enc,
                                              key_masks=src_masks,
                                              num_heads=self.hp.num_heads,
                                              dropout_rate=self.hp.dropout_rate,
                                              training=training,
                                              causality=False)
                    # feed forward
                    enc = ff(enc, num_units=[self.hp.d_ff, self.hp.d_model])
            
            if self.hp.pro:
                ## Blocks
                for i in range(self.hp.num_blocks):
                    with tf.variable_scope("cxt_num_blocks_{}".format(i), reuse=tf.AUTO_REUSE):
                        # self-attention
                        cxt_enc = multihead_attention(queries=cxt_enc,
                                                keys=cxt_enc,
                                                values=cxt_enc,
                                                key_masks=pro_src_masks,
                                                num_heads=self.hp.num_heads,
                                                dropout_rate=self.hp.dropout_rate,
                                                training=training,
                                                causality=False)
                        # feed forward
                        cxt_enc = ff(cxt_enc, num_units=[self.hp.d_ff, self.hp.d_model]) #[bz, cxt_len, d_model]
        
        
        if self.hp.pro:
            call_weights = tf.nn.softmax(1 + self.invoked_batch)
            cxt_enc = tf.multiply(tf.expand_dims(call_weights, -1), cxt_enc)
            memory = enc, cxt_enc
            masks = src_masks, pro_src_masks
        else:
            memory = enc, None
            masks = src_masks, None

        # memory = enc
        return memory, masks

    def decode(self, ys, memorys, masks, training=True):
        '''
        memory: encoder outputs. (N, T1, d_model)
        src_masks: (N, T1)
        Returns
        logits: (N, T2, V). float32.
        y_hat: (N, T2). int32
        y: (N, T2). int32
        sents2: (N,). string.
        '''
        memory, pro_memory = memorys
        src_masks, pro_src_masks = masks
        with tf.variable_scope("decoder", reuse=tf.AUTO_REUSE):
            decoder_inputs, y = ys
            # decoder_inputs, y = self.dec_inp_batch, self.dec_tgt_batch

            # tgt_masks
            tgt_masks = tf.math.equal(decoder_inputs, self.data.PAD)  # (N, T2)

            # embedding
            dec = tf.nn.embedding_lookup(self.embeddings, decoder_inputs)  # (N, T2, d_model)
            dec *= self.hp.d_model ** 0.5  # scale

            dec += positional_encoding(dec, self.data.tgt_name_len)
            dec = tf.layers.dropout(dec, self.hp.dropout_rate, training=training)

            # Blocks
            for i in range(self.hp.num_blocks):
                with tf.variable_scope("num_blocks_{}".format(i), reuse=tf.AUTO_REUSE):
                    # Masked self-attention (Note that causality is True at this time)
                    dec = multihead_attention(queries=dec,
                                              keys=dec,
                                              values=dec,
                                              key_masks=tgt_masks,
                                              num_heads=self.hp.num_heads,
                                              dropout_rate=self.hp.dropout_rate,
                                              training=training,
                                              causality=True,
                                              scope="self_attention")
                    if self.hp.pro:
                        # Vanilla attention with project cxt
                        dec1 = multihead_attention(queries=dec,
                                                keys=pro_memory,
                                                values=pro_memory,
                                                key_masks=pro_src_masks,
                                                num_heads=self.hp.num_heads,
                                                dropout_rate=self.hp.dropout_rate,
                                                training=training,
                                                causality=False,
                                                scope="vanilla_attention")
                        
                        # Vanilla attention
                        dec2 = multihead_attention(queries=dec1,
                                                keys=memory,
                                                values=memory,
                                                key_masks=src_masks,
                                                num_heads=self.hp.num_heads,
                                                dropout_rate=self.hp.dropout_rate,
                                                training=training,
                                                causality=False,
                                                scope="vanilla_attention")
                        dec = dec2
                    else:
                        # Vanilla attention
                        dec = multihead_attention(queries=dec,
                                                keys=memory,
                                                values=memory,
                                                key_masks=src_masks,
                                                num_heads=self.hp.num_heads,
                                                dropout_rate=self.hp.dropout_rate,
                                                training=training,
                                                causality=False,
                                                scope="vanilla_attention")
                    ### Feed Forward
                    dec = ff(dec, num_units=[self.hp.d_ff, self.hp.d_model])

        # Final linear projection (embedding weights are shared)
        weights = tf.transpose(self.embeddings) # (d_model, vocab_size)
        logits = tf.einsum('ntd,dk->ntk', dec, weights) # (N, T2, vocab_size)
        y_hat = tf.to_int32(tf.argmax(logits, axis=-1))

        return logits, y_hat, y

    def train(self):
        '''
        Returns
        loss: scalar.
        train_op: training operation
        global_step: scalar.
        summaries: training summary node
        '''
        # forward
        memory, src_masks = self.encode()
        ys = self.dec_inp_batch, self.dec_tgt_batch
        logits, preds, y = self.decode(ys, memory, src_masks)

        # train scheme
        y_ = label_smoothing(tf.one_hot(y, depth=len(self.data.w2id)))
        ce = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y_)
        nonpadding = tf.to_float(tf.not_equal(y, self.data.PAD))  # 0: <pad>
        loss = tf.reduce_sum(ce * nonpadding) / (tf.reduce_sum(nonpadding) + 1e-7)

        global_step = tf.train.get_or_create_global_step()
        lr = noam_scheme(self.hp.lr, global_step, self.hp.warmup_steps)
        optimizer = tf.train.AdamOptimizer(lr)
        train_op = optimizer.minimize(loss, global_step=global_step)

        tf.summary.scalar('lr', lr)
        tf.summary.scalar("loss", loss)
        tf.summary.scalar("global_step", global_step)

        summaries = tf.summary.merge_all()

        return loss, train_op, global_step, summaries, preds

    def eval(self):
        '''Predicts autoregressively
        At inference, input ys is ignored.
        Returns
        y_hat: (N, T2)
        '''
        # decoder_inputs, y = ys

        decoder_inputs = tf.ones((tf.shape(self.body_batch)[0], 1), tf.int32) * self.data.BOS
        ys = (decoder_inputs, self.dec_tgt_batch)

        memory, src_masks = self.encode(False)

        logging.info("Inference graph is being built. Please be patient.")
        for _ in range(self.data.tgt_name_len):
            logits, y_hat, y = self.decode(ys, memory, src_masks, False)
            if tf.reduce_sum(y_hat, 1) == self.data.PAD: break

            _decoder_inputs = tf.concat((decoder_inputs, y_hat), 1)
            ys = (_decoder_inputs, y)

        # monitor a random sample
        # n = tf.random_uniform((), 0, tf.shape(y_hat)[0]-1, tf.int32)
        # sent1 = sents1[n]
        # pred = convert_idx_to_token_tensor(y_hat[n], self.idx2token)
        # sent2 = sents2[n]
        #
        # tf.summary.text("sent1", sent1)
        # tf.summary.text("pred", pred)
        # tf.summary.text("sent2", sent2)
        # summaries = tf.summary.merge_all()

        # return y_hat, summaries
        return y_hat
