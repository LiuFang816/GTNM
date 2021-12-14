# -*- coding: utf-8 -*-
#/usr/bin/python3
import sys 
sys.path.append("../data_processing/") 

import tensorflow as tf
# from new_data_loader import *
from model_invoked import Transformer
from extract_data_subword import *
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


def test():
    resout = open(outfile, 'a')
    
    logging.info("# Load model")
    m = Transformer(hp)
    y_hat = m.eval()

    logging.info("# Session")
    saver = tf.train.Saver(max_to_keep=hp.save_epochs)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        ckpt = tf.train.latest_checkpoint(hp.logdir)
        saver.restore(sess, ckpt)

        summary_writer = tf.summary.FileWriter(hp.logdir, sess.graph)

        # test
        test_hypotheses, test_precision, test_recall, test_f1, test_acc = get_hypotheses('test_subword', hp, sess, m, y_hat, m.data.w2id, m.data.id2w)
        print('test precision {}, test recall {}, test f1 {}, test acc {}'.format(test_precision, test_recall, test_f1, test_acc), file=resout)
        resout.flush()
    
    summary_writer.close()
    resout.close()

def main(_):
    test()

if __name__ == '__main__':
    tf.app.run()


