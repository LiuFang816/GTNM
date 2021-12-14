# -*- coding: utf-8 -*-
#/usr/bin/python3
import sys 
sys.path.append("../data_processing/") 

import tensorflow as tf
# from new_data_loader import *
from model_invoked import Transformer
from  extract_data_subword import *
from utils import save_hparams, save_variable_specs, get_hypotheses, calc_bleu
import os
from hparams import Hparams
import math
import logging

logging.basicConfig(level=logging.INFO)

logging.info("# hparams")
hparams = Hparams()
parser = hparams.parser
hp = parser.parse_args()
save_hparams(hp, hp.logdir)
os.environ['CUDA_VISIBLE_DEVICES'] = hp.gpu
outfile = hp.res_log

def run_epoch(session, model, state, summary_writer, epoch=None):
    total_loss = 0.0

    data_loader = model.data.batch_iter(hp.batch_size, state, epoch=epoch)
    step = 0
    while True:
        feed_dict = {}

        body_batch, pro_batch, doc_batch, dec_inp_batch, dec_tgt_batch, invoked_batch, batch_len = next(data_loader)
        feed_dict[model.body_batch] = body_batch
        feed_dict[model.pro_batch] = pro_batch
        feed_dict[model.doc_batch] = doc_batch
        feed_dict[model.invoked_batch] = invoked_batch
        feed_dict[model.dec_inp_batch] = dec_inp_batch
        feed_dict[model.dec_tgt_batch] = dec_tgt_batch

        _, _gs, _summary, _loss, _preds = session.run([model.train_op, model.global_step, model.train_summaries, model.loss, model.preds], feed_dict)
        summary_writer.add_summary(_summary, _gs)

        if step % (batch_len // 10) == 10:
        # if step % 10 == 0:
            print("%.2f perplexity : %.3f " %
                  (step * 1.0 / batch_len, _loss))
            # print("%.2f perplexity : %.3f " %
            #       (step * 1.0 / batch_len, _loss))

        total_loss += _loss
        step += 1
        if step >= batch_len:
            break
    return total_loss / batch_len, _gs, _preds, dec_tgt_batch


def train():
    resout = open(outfile, 'a')
    # data.read_raw_data()
    # data.save_data()
    # data.load_data()

    logging.info("# Load model")
    m = Transformer(hp)
    m.loss, m.train_op, m.global_step, m.train_summaries, m.preds = m.train()
    y_hat = m.eval()

    logging.info("# Session")
    saver = tf.train.Saver(max_to_keep=hp.save_epochs)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        ckpt = tf.train.latest_checkpoint(hp.logdir)
        if ckpt is None:
            logging.info("Initializing from scratch")
            sess.run(tf.global_variables_initializer())
            save_variable_specs(os.path.join(hp.logdir, "specs"))
        else:
            saver.restore(sess, ckpt)

        summary_writer = tf.summary.FileWriter(hp.logdir, sess.graph)
        best_eval_val_f1 = 0
        for i in range(hp.num_epochs):
            
            train_loss, _global_step, _preds, _tgt = run_epoch(sess, m, 'train_subword0', summary_writer, epoch=i)
            _, train_precision, train_recall, train_f1, train_acc = get_hypotheses('train_subword0', hp, sess, m, y_hat, m.data.w2id, m.data.id2w, epoch=i)
            print('epoch {}: train precision {}, train recall {}, train f1 {}, train acc {}'.format(i+1, train_precision, train_recall, train_f1, train_acc))
            

            logging.info("# valiation")

            logging.info("# get hypotheses")

            hypotheses, val_precision, val_recall, val_f1, val_acc = get_hypotheses('eval_subword', hp, sess, m, y_hat, m.data.w2id, m.data.id2w)
            print('epoch {}: eval precision {}, eval recall {}, eval f1 {}, eval acc {}'.format(i+1, val_precision, val_recall, val_f1, val_acc), file=resout)
            resout.flush()
            # print(hypotheses)
            if val_f1 > best_eval_val_f1:
                best_eval_val_f1 = val_f1
                logging.info("# write eval results")
                model_output = "java_E%02dL%.2f" % (i, train_loss)
        
                logging.info("# save models")
                ckpt_name = os.path.join(hp.logdir, model_output)
                saver.save(sess, ckpt_name, global_step=_global_step)
                logging.info("after training of {} epochs, {} has been saved.".format(i, ckpt_name))


            logging.info("# fall back to train mode")
            # sess.run(train_init_op)

    summary_writer.close()
    resout.close()

def main(_):
    train()

if __name__ == '__main__':
    tf.app.run()


