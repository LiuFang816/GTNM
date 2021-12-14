# -*- coding: utf-8 -*-
# /usr/bin/python3
'''
Feb. 2019 by kyubyong park.
kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/transformer.
Utility functions
'''

import tensorflow as tf
# from tensorflow.python import pywrap_tensorflow
# import numpy as np
import json
import os, re
import logging

logging.basicConfig(level=logging.INFO)
# word2id = json.loads(open('sub_token_w2id.txt','r').read())

def str_split(str):
    if str == '':
        return ['']
    if '_' in str:
        return [s.lower() for s in str.split('_')]
    words = [[str[0].lower()]]

    for c in str[1:]:
        if words[-1][-1].islower() and c.isupper():
            words.append(list(c.lower()))
        else:
            words[-1].append(c)

    return [''.join(word) for word in words]

def calc_num_batches(total_num, batch_size):
    '''Calculates the number of batches.
    total_num: total sample number
    batch_size
    Returns
    number of batches, allowing for remainders.'''
    return total_num // batch_size + int(total_num % batch_size != 0)

def convert_idx_to_token_tensor(inputs, idx2token):
    '''Converts int32 tensor to string tensor.
    inputs: 1d int32 tensor. indices.
    idx2token: dictionary
    Returns
    1d string tensor.
    '''
    def my_func(inputs):
        return " ".join(idx2token[elem] for elem in inputs)

    return tf.py_func(my_func, [inputs], tf.string)

# # def pad(x, maxlen):
# #     '''Pads x, list of sequences, and make it as a numpy array.
# #     x: list of sequences. e.g., [[2, 3, 4], [5, 6, 7, 8, 9], ...]
# #     maxlen: scalar
# #
# #     Returns
# #     numpy int32 array of (len(x), maxlen)
# #     '''
# #     padded = []
# #     for seq in x:
# #         seq += [0] * (maxlen - len(seq))
# #         padded.append(seq)
# #
# #     arry = np.array(padded, np.int32)
# #     assert arry.shape == (len(x), maxlen), "Failed to make an array"
#
#     return arry


def postprocess(hypotheses, idx2token):
    '''Processes translation outputs.
    hypotheses: list of encoded predictions
    idx2token: dictionary
    Returns
    processed hypotheses
    '''
    _hypotheses = []
    for h in hypotheses:
        sent = [idx2token[idx] for idx in h]
        sent = sent[:sent.index("EOS")] if "EOS" in sent else sent
        # sent = sent.replace("▁", " ") # remove bpe symbols
        _hypotheses.append(sent)
    return _hypotheses


def save_hparams(hparams, path):
    '''Saves hparams to path
    hparams: argsparse object.
    path: output directory.
    Writes
    hparams as literal dictionary to path.
    '''
    if not os.path.exists(path): os.makedirs(path)
    hp = json.dumps(vars(hparams))
    with open(os.path.join(path, "hparams"), 'w') as fout:
        fout.write(hp)

def load_hparams(parser, path):
    '''Loads hparams and overrides parser
    parser: argsparse parser
    path: directory or file where hparams are saved
    '''
    if not os.path.isdir(path):
        path = os.path.dirname(path)
    d = open(os.path.join(path, "hparams"), 'r').read()
    flag2val = json.loads(d)
    for f, v in flag2val.items():
        parser.f = v

def save_variable_specs(fpath):
    '''Saves information about variables such as
    their name, shape, and total parameter number
    fpath: string. output file path
    Writes
    a text file named fpath.
    '''
    def _get_size(shp):
        '''Gets size of tensor shape
        shp: TensorShape
        Returns
        size
        '''
        size = 1
        for d in range(len(shp)):
            size *=shp[d]
        return size

    params, num_params = [], 0
    for v in tf.global_variables():
        params.append("{}==={}".format(v.name, v.shape))
        num_params += _get_size(v.shape)
    print("num_params: ", num_params)
    with open(fpath, 'w') as fout:
        fout.write("num_params: {}\n".format(num_params))
        fout.write("\n".join(params))
    logging.info("Variables info has been saved.")


def get_acc(preds, targets, w2id):
    total = 0.0
    correct = 0.0
    for i in range(len(targets)):
        total += 1
        pre = preds[i][:preds[i].index(w2id["EOS"])] if w2id["EOS"] in preds[i] else preds[i]
        tar = targets[i][:targets[i].index(w2id["EOS"])] if w2id["EOS"] in targets[i] else targets[i]
        if pre == tar:
            correct += 1

    return correct/total

def update_per_subtoken_statistics(results, true_positive, false_positive, false_negative):
    for original_name, predicted in results:
        filtered_predicted_names = predicted
        filtered_original_subtokens = original_name
        # print("original_name: {}, predicted: {}".format(''.join(original_name), ''.join(predicted)))
        if ''.join(filtered_original_subtokens) == ''.join(filtered_predicted_names):
            true_positive += len(filtered_original_subtokens)
            continue

        for subtok in filtered_predicted_names:
            if subtok in filtered_original_subtokens:
                true_positive += 1
            else:
                false_positive += 1
        for subtok in filtered_original_subtokens:
            if not subtok in filtered_predicted_names:
                false_negative += 1


    return true_positive, false_positive, false_negative


def calculate_results(true_positive, false_positive, false_negative):
    if true_positive + false_positive > 0:
        precision = true_positive / (true_positive + false_positive)
    else:
        precision = 0
    if true_positive + false_negative > 0:
        recall = true_positive / (true_positive + false_negative)
    else:
        recall = 0
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0
    return precision, recall, f1

def get_hypotheses(state, hp, sess, model, tensor, w2id, id2w, epoch=None):
    '''Gets hypotheses.
    num_batches: scalar.
    num_samples: scalar.
    sess: tensorflow sess object
    tensor: target tensor to fetch
    dict: idx2token dictionary
    Returns
    hypotheses: list of sents
    '''

    hypotheses = []
    tgt = []
    data_loader = model.data.batch_iter(hp.batch_size, state, epoch)
    step = 0
    true_positive = 0
    false_positive = 0
    false_negative = 0
    while True:
        feed_dict = {}
        body_batch, pro_batch, doc_batch, dec_inp_batch, dec_tgt_batch, invoked_batch, batch_len = next(data_loader)
        # print(cxt_names_batch)
        feed_dict[model.body_batch] = body_batch
        feed_dict[model.pro_batch] = pro_batch
        feed_dict[model.doc_batch] = doc_batch
        feed_dict[model.dec_inp_batch] = dec_inp_batch
        feed_dict[model.dec_tgt_batch] = dec_tgt_batch
        feed_dict[model.invoked_batch] = invoked_batch
        h = sess.run(tensor, feed_dict)
        hypotheses.extend(h.tolist())
        tgt.extend(dec_tgt_batch.tolist())

        true_positive, false_positive, false_negative = update_per_subtoken_statistics(
            zip(postprocess(dec_tgt_batch.tolist(), id2w), postprocess(h.tolist(),id2w)),
            true_positive, false_positive, false_negative)
        # precision, recall, f1 = calculate_results(true_positive, false_positive, false_negative)
        # print("precision: {}, recall: {}, f1: {}".format(precision, recall, f1))

        # print(h.tolist())
        # print()
        # print(dec_tgt_batch)
        # print('====')

        step += 1
        if step >= batch_len:
            break
    acc = get_acc(hypotheses, tgt, w2id)
    precision, recall, f1 = calculate_results(true_positive, false_positive, false_negative)
    print("precision: {}, recall: {}, f1: {}, exact match: {}".format(precision, recall, f1, acc))
    hypotheses = postprocess(hypotheses, id2w)
    # print(hypotheses)
    return hypotheses, precision, recall, f1, acc

def calc_bleu(ref, translation):
    '''Calculates bleu score and appends the report to translation
    ref: reference file path
    translation: model output file path
    Returns
    translation that the bleu score is appended to'''
    get_bleu_score = "perl multi-bleu.perl {} < {} > {}".format(ref, translation, "temp")
    os.system(get_bleu_score)
    bleu_score_report = open("temp", "r").read()
    with open(translation, "a") as fout:
        fout.write("\n{}".format(bleu_score_report))
    try:
        score = re.findall("BLEU = ([^,]+)", bleu_score_report)[0]
        new_translation = translation + "B{}".format(score)
        os.system("mv {} {}".format(translation, new_translation))
        os.remove(translation)

    except: pass
    os.remove("temp")


    # def get_inference_variables(ckpt, filter):
    #     reader = pywrap_tensorflow.NewCheckpointReader(ckpt)
    #     var_to_shape_map = reader.get_variable_to_shape_map()
    #     vars = [v for v in sorted(var_to_shape_map) if filter not in v]
    #     return vars
