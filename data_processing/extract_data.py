"""
Extract dataset with localness info in sequence format
"""
import time
import signal 
import json
import re
import sentencepiece as spm
import numpy as np
from collections import OrderedDict
from pathos import multiprocessing
from normalizer import Normalizer
import os
import argparse
import logging
import pickle
from tqdm import tqdm
from fuzzywuzzy import fuzz


def set_timeout(num, callback):
    def wrap(func):
        def handle(signum, frame):  
            raise RuntimeError

        def to_do(*args, **kwargs):
            try:
                signal.signal(signal.SIGALRM, handle)  
                signal.alarm(num)  
                r = func(*args, **kwargs)
                signal.alarm(0)  
                return r
            except RuntimeError as e:
                callback()

        return to_do

    return wrap

def after_timeout():  
    print("Time out!")
    return None


normalizer = Normalizer("java")
def str_split(str):
    if '_' in str:
        return [s.lower() for s in str.split('_')]
    words = [[str[0].lower()]]

    for c in str[1:]:
        if words[-1][-1].islower() and c.isupper():
            words.append(list(c.lower()))
        else:
            words[-1].append(c)

    return [''.join(word) for word in words]
    
def travel_data(data):
    for file in data["files"]:
        yield file
    for subdir in data["subdirs"]:
        for datax in travel_data(subdir):
            yield datax
def read_as_pkl(filename):
    data = pickle.load(open(filename, "rb"))
    for i, project_data in enumerate(data):
        if i % 100 == 0:
            print(i)
        for datax in travel_data(project_data):
            yield datax

def update_vocab(vocab_path):
    with open(vocab_path, 'r') as f:
        word2id = json.loads(f.read())
        id2word = {v: k for k, v in word2id.items()}
        word2id["<endofline>"] = 4
        word2id["<endoftext>"] = 5
        for word in word2id.keys():
            if word not in ['PAD', 'BOS', 'EOS', 'UNK', "<endofline>", "<endoftext>"]:
                word2id[word] += 2
        id2word = {v: k for k, v in word2id.items()}
   
    with open(vocab_path, 'w') as f:
        f.write(json.dumps(word2id))
    return word2id, id2word
        

def load_vocab(vocab_path):
    with open(vocab_path, 'r') as f:
        word2id = json.loads(f.read())
        id2word = {v: k for k, v in word2id.items()}
    return word2id, id2word

def data_to_id(data, word_to_id):
    data = data.split()
    sub_words = []
    for item in data:
        sub_words.extend(str_split(item))
    new_data = []
    for i in range(len(sub_words)):
        new_data.append(word_to_id[sub_words[i]] if sub_words[i] in word_to_id else word_to_id['UNK'])
    return new_data

def doc_data_to_id(data, word_to_id):
    data = data.split()
    new_data = []
    for i in range(len(data)):
        data[i] = data[i].lower()
        new_data.append(word_to_id[data[i]] if data[i] in word_to_id else word_to_id['UNK'])
    return new_data

class localContext(object):
    def __init__(self, body_context_size, doc_context_size, project_context_size, tgt_name_size, sub_vocab_file, doc_vocab_file, include_docstring=False, expr_max_len=1024, expr_max_num=30, custom_eol="<endofline>", custom_eot="<endoftext>", custom_bos="BOS", custom_eos="EOS", custom_pad="PAD", datapath=None):
        self.body_context_size = body_context_size
        self.doc_context_size = doc_context_size
        self.project_context_size = project_context_size
        self.tgt_name_len = tgt_name_size
        # self.sp_model = spm.SentencePieceProcessor()
        # self.sp_model.load(vocab_file)
        self.w2id, self.id2w = load_vocab(sub_vocab_file) 
        self.doc_w2id, self.doc_id2w = load_vocab(doc_vocab_file) 
        self.docstring = include_docstring
        self.expr_max_len = expr_max_len
        self.expr_max_num = expr_max_num
        self.END_OF_LINE = self.w2id[custom_eol]
        self.END_OF_TEXT = self.w2id[custom_eot]
        self.BOS = self.w2id[custom_bos]
        self.EOS = self.w2id[custom_eos]
        self.PAD = self.w2id[custom_pad]
        self.datapath = datapath

    def read_as_pkl(self, filename):
        data = pickle.load(open(filename, "rb"))
        for i, project_data in enumerate(data):
            for datax in self.travel_data(project_data):
                yield datax
           
    def travel_data(self, data):    
        for file in data["files"]:
            yield file
        for subdir in data["subdirs"]:
            for datax in self.travel_data(subdir):
                yield datax

    def encode_for_sp_old(self, code_before):
        codes = normalizer.normalize(code_before)
        if not len(codes):
            return []
        initial_tokens = []
        for code_segment in codes.splitlines():
            initial_tokens.extend(self.sp_model.EncodeAsIds(code_segment))
            initial_tokens.append(self.END_OF_LINE)
        if code_before[-1] != "\n":
            initial_tokens.pop(-1)

        return initial_tokens 
    
    def encode_for_sp(self, code_before, vocab):
        codes = normalizer.normalize(code_before)
        if not len(codes):
            return []
        initial_tokens = []
        for code_segment in codes.splitlines():
            initial_tokens.extend(data_to_id(code_segment, vocab))
            initial_tokens.append(self.END_OF_LINE)
        if code_before[-1] != "\n":
            initial_tokens.pop(-1)

        return initial_tokens 


    def process_functions(self, data, localness):
        """
        Iterate over all functions, extract signatures, bodies and roll a window
        """
        samples = []
        tags = []
        for method in data["schema"]["methods"]:
            st, _ = method["byte_span"]
            encoded_other_method = []
            best_edit_sim_name = 60
            best_other_method_idx = -1
            for i, other in enumerate(data["schema"]["methods"]):
                if other["byte_span"][0] < st:
                    if fuzz.ratio(method["name"], other["name"]) > best_edit_sim_name:
                        best_edit_sim_name = fuzz.ratio(method["name"], other["name"])
                        best_other_method_idx = i
            for i, other in enumerate(data["schema"]["methods"]):
                if other["byte_span"][0] < st:
                    encoded_other_method.extend(self.encode_for_sp(other["signature"], self.w2id))
                    encoded_other_method.append(self.END_OF_LINE)
                    if i == best_other_method_idx:
                        encoded_other_body = self.encode_for_sp(other["body"], self.w2id)
                        n_body = len(encoded_other_body)
                        pos = n_body
                        if n_body > self.local_context_size//2:
                            pos = self.local_context_size//2
                            while pos < n_body and encoded_other_body[pos] != self.END_OF_LINE:
                                pos += 1
                        encoded_other_method.extend(encoded_other_body[:pos])
                        encoded_other_method.append(self.END_OF_LINE)
            encoded_body = self.encode_for_sp(method["body"], self.w2id)
            encoded_signature = self.encode_for_sp(method["signature"], self.w2id) + [self.END_OF_LINE]
            encoded_docstring = self.encode_for_sp(method["docstring"], self.doc_w2id)
            encoded_docstring += [self.END_OF_LINE] if len(encoded_docstring) else []
            num_context_windows = len(encoded_body) // self.context_size + 1
            if self.docstring:
                local_context = (localness + encoded_other_method + [self.END_OF_TEXT] + encoded_signature + encoded_docstring)[-self.body_context_size :]
            else:
                local_context = (localness + encoded_other_method + [self.END_OF_TEXT] + encoded_signature)[-self.body_context_size :]
            for icontext_window in range(num_context_windows):
                samples.append(
                    local_context
                    + encoded_body[
                        icontext_window
                        * self.context_size : (icontext_window + 1)
                        * self.context_size
                    ]
                )
                tags.append(
                    [0] * len(local_context)
                    + [1] * min(self.context_size, len(encoded_body)-icontext_window*self.context_size)
                )
                assert len(samples[-1]) == len(tags[-1])

        return samples, tags

    def process_class_methods(self, clazz, localness):
        """
        Note: consider combining with process_functions, only difference is need to add class level context to local context
        """
        samples = []
        tags = []
        for method in clazz["methods"]:
            st, _ = method["byte_span"]
            encoded_other_method = []
            best_edit_sim_name = 60
            best_other_method_idx = -1
            for i, other in enumerate(clazz["methods"]):
                if other["byte_span"][0] < st:
                    if fuzz.ratio(method["name"], other["name"]) > best_edit_sim_name:
                        best_edit_sim_name = fuzz.ratio(method["name"], other["name"])
                        best_other_method_idx = i
            for i, other in enumerate(clazz["methods"]):
                if other["byte_span"][0] < st:
                    encoded_other_method.extend(self.encode_for_sp(other["signature"], self.w2id))
                    encoded_other_method.append(self.END_OF_LINE)
                    if i == best_other_method_idx:
                        encoded_other_body = self.encode_for_sp(other["body"], self.w2id)
                        n_body = len(encoded_other_body)
                        pos = n_body
                        if n_body > self.local_context_size//2:
                            pos = self.local_context_size//2
                            while pos < n_body and encoded_other_body[pos] != self.END_OF_LINE:
                                pos += 1
                        encoded_other_method.extend(encoded_other_body[:pos])
                        encoded_other_method.append(self.END_OF_LINE)
            encoded_body = self.encode_for_sp(method["body"], self.w2id)
            encoded_signature = self.encode_for_sp(method["signature"], self.w2id) + [self.END_OF_LINE]
            encoded_docstring = self.encode_for_sp(method["docstring"], self.doc_w2id)
            encoded_docstring += [self.END_OF_LINE] if len(encoded_docstring) else []
            num_context_windows = len(encoded_body) // self.context_size + 1
            if self.docstring:
                local_context = (localness + encoded_other_method + [self.END_OF_TEXT] + encoded_signature + encoded_docstring)[-self.local_context_size :]
            else:
                local_context = (localness + encoded_other_method + [self.END_OF_TEXT] + encoded_signature)[-self.local_context_size :]
            for icontext_window in range(num_context_windows):
                samples.append(
                    local_context
                    + encoded_body[
                        icontext_window
                        * self.context_size : (icontext_window + 1)
                        * self.context_size
                    ]
                )
                tags.append(
                    [0] * len(local_context)
                    + [1] * min(self.context_size, len(encoded_body)-icontext_window*self.context_size)
                )
                assert len(samples[-1]) == len(tags[-1])

        return samples, tags
    
    def process_class_methods_for_method_nameing(self, clazz, project_specific_context, encoded_import_context=None):
        """
        Note: consider combining with process_functions, only difference is need to add class level context to local context
        """
    
        project_level_cxt = []
        body = []
        doc = []

        tags = []
        overlap = 0 
        # print(len(clazz["methods"]))
        for method in clazz["methods"]:
            st, _ = method["byte_span"]
            encoded_other_method = []
            best_edit_sim_name = 60
            best_other_method_idx = -1
            
            for i, other in enumerate(clazz["methods"]):
                if other["byte_span"][0] != st:
                    encoded_other_method.extend(self.encode_for_sp(other["signature"], self.w2id))
                    encoded_other_method.append(self.END_OF_LINE)
                    
            encoded_body = self.encode_for_sp(method["body"], self.w2id)
            encoded_identifiers =self.encode_for_sp(method["identifiers"], self.w2id)
    
            encoded_signature = self.encode_for_sp(method["signature_woname"], self.w2id) + [self.END_OF_LINE]
            encoded_name = self.encode_for_sp(method["name"], self.w2id)
            encoded_docstring = self.encode_for_sp(method["docstring"], self.doc_w2id)
            encoded_docstring += [self.END_OF_LINE] if len(encoded_docstring) else []
            
            body.append((encoded_signature + encoded_identifiers)[-self.body_context_size :])
            if self.docstring:
                doc.append(encoded_docstring[self.doc_context_size:])
        
            project_level_cxt.append((project_specific_context + encoded_other_method)[-self.project_context_size :])
            
            tags.append(encoded_name)
    
            overlap += len(list(set(encoded_name).intersection(set(encoded_import_context))))
        return body, project_level_cxt, doc, tags, overlap

    @set_timeout(60, after_timeout)
    def process_one_file(self, data):
        body_samples = []
        doc_samples = []
        project_cxt_samples = []
        tags = []
    
        encoded_import_context = []
        #print(data["imports"])
        for import_context in data["imports"]:
            if "methods" in import_context:  # method
                methods = []
                for m in import_context["methods"]:
                    methods.append(m['name'])
                encoded_import_context.extend(self.encode_for_sp(' '.join(methods), self.w2id) + [self.END_OF_LINE])
            elif "definition" in import_context:   # class
                encoded_import_context.extend(self.encode_for_sp(import_context["definition"], self.w2id) + [self.END_OF_LINE])
            # elif "signature" in import_context:     # method
            #     encoded_import_context.extend(self.encode_for_sp(import_context["signature"]) + [self.END_OF_LINE])

        encoded_file_level_context = self.encode_for_sp(
            "\n".join([context for context in data["schema"]["contexts"][:self.expr_max_num] if len(context) < self.expr_max_len])
        , self.w2id) + [self.END_OF_LINE]

        total_methods = 0
        total_overlap = 0
        for clazz in data["schema"]["classes"]:
            st, _ = clazz["byte_span"]
            encoded_other_method = []
            for other in data["schema"]["methods"]:
                if other["byte_span"][0] < st:
                    encoded_other_method.extend(self.encode_for_sp(other["signature"], self.w2id))
                    encoded_other_method.append(self.END_OF_LINE)

            encoded_class_signature = self.encode_for_sp(clazz["definition"], self.w2id) + [self.END_OF_LINE]
            try:
                encoded_class_globals = self.encode_for_sp(
                    "\n".join([attr for attr in clazz["attributes"]["attribute_expressions"][:self.expr_max_num] if len(attr) < self.expr_max_len]), self.w2id
                ) + [self.END_OF_LINE]
            except KeyError:
                encoded_class_globals = []
            encoded_class_level_context = (
                encoded_class_signature + encoded_class_globals
            )
            # ext_sample, ext_tag = self.process_class_methods_for_method_nameing(
            #     clazz, encoded_import_context+encoded_file_level_context+encoded_other_method+encoded_class_level_context
            # )

            project_specific_context = encoded_import_context+encoded_file_level_context+encoded_class_level_context
            body_sample, project_context_sample, doc_sample, method_names, overlap = self.process_class_methods_for_method_nameing(clazz, project_specific_context, encoded_import_context)
          
            total_methods += len(method_names)
            total_overlap += overlap
            body_samples.extend(body_sample)
            doc_samples.extend(doc_sample)
            project_cxt_samples.extend(project_context_sample)
            tags.extend(method_names)

        return body_samples, project_cxt_samples, doc_samples, tags, total_methods, total_overlap, len(encoded_import_context)
    
    def extract_samples(self, ifilename, ofilename):
        body_samples = []
        doc_samples = []
        project_cxt_samples = []
        tags = []
       
        only_in_doc = 0
        only_in_project_cxt = 0

        # pool = multiprocessing.Pool(processes=16)
        # for data in pool.imap(self.process_one_file, self.read_as_pkl(ifilename), chunksize=100):
        #     samples.extend(data)
        count = 0
        total_methods = 0
        total_overlap = 0

        cross_file_cxt_len = []
        for data in self.read_as_pkl(ifilename):
            count += 1
            if count % 1000 == 0:
                print(count)
            info = self.process_one_file(data)
            if not info:
                continue
            else:
                body_sample, project_cxt_sample, doc_cxt_sample, tag, _total_methods, _total_overlap, cross_file_len = info
            
            # for i in range(len(tag)):
            #     if not list(set(tag[i])&set(body_sample[i])):
            #         if doc_cxt_sample and list(set(tag[i])&set(doc_cxt_sample[i])):
            #             only_in_doc += 1
            #         if project_cxt_sample and list(set(tag[i])&set(project_cxt_sample[i])):
            #             only_in_project_cxt += 1

            if cross_file_len > 0:
                cross_file_cxt_len.append(cross_file_len)
            total_methods += _total_methods
            total_overlap += _total_overlap
            body_samples.extend(body_sample)
            doc_samples.extend(doc_cxt_sample)
            project_cxt_samples.extend(project_cxt_sample)
            tags.extend(tag)
        
        #print("total methods: {} total overlap:{} proprotion: {}".format(total_methods, total_overlap, total_overlap/total_methods))
        # print("avg cross file methods len: {}, med cross file methods len: {}".format(np.average(cross_file_cxt_len), np.median(cross_file_cxt_len)))

        print("writing...")
        pickle.dump(body_samples, open(ofilename+"_body.pkl", "wb"))
        pickle.dump(doc_samples, open(ofilename+"_doc.pkl", "wb"))
        pickle.dump(project_cxt_samples, open(ofilename+"_pro.pkl", "wb"))
        pickle.dump(tags, open(ofilename+"_tag.pkl", "wb"))
    
    def process_normal_test(self, data):
        file_content = data["content"]
        intervals = []
        for method in data["schema"]["methods"]:
            interval = None
            st = 0
            while interval is None or interval in intervals: 
                pos = file_content[st:].find(method["original_string"].lstrip())
                assert pos != -1, file_content+"\n"+"="*50+"\n"+method["original_string"].lstrip()
                inner_pos = method["original_string"].lstrip().find(method["body"].lstrip())
                assert inner_pos != -1
                pos += st + inner_pos
                interval = [pos, pos+len(method["body"].lstrip())]
                st = interval[1]
            intervals.append(interval)
        
        for clazz in data["schema"]["classes"]:
            clazz_st = file_content.find(clazz["original_string"].lstrip())
            clazz_end = clazz_st + len(clazz["original_string"].lstrip())
            for method in clazz["methods"]:
                interval = None
                st = clazz_st
                while interval is None or interval in intervals: 
                    pos = file_content[st:clazz_end].find(method["original_string"].lstrip())
                    if pos == -1 and interval is not None:
                        break
                    assert pos != -1, file_content[st:clazz_end]+"\n"+"="*50+"\n"+method["original_string"].lstrip()
                    inner_pos = method["original_string"].lstrip().find(method["body"].lstrip())
                    assert inner_pos != -1
                    pos += st + inner_pos
                    interval = [pos, pos+len(method["body"].lstrip())]
                    st = interval[1]
                if interval not in intervals:
                    intervals.append(interval)
        return sample, tag

    def get_dec_inp_targ_seqs(self, sequence, max_len):
        dec_inp = []
        dec_tgt = []
        for line in sequence:
            inp = [self.BOS] + line[:]
            target = line[:]
            if len(inp) > max_len: # truncate
                inp = inp[:max_len]
                target = target[:max_len] # no end_token
            else: # no truncation
                target.append(self.EOS) # end token
            assert len(inp) == len(target)
            dec_inp.append(inp)
            dec_tgt.append(target)
        return dec_inp, dec_tgt

    def read_results(self, filename):

        def convert_to_normal(sample):
            cur_stat = []
            for x in sample:
                if x in [self.END_OF_LINE, self.END_OF_TEXT]:
                    if cur_stat:
                        # print(cur_stat)
                        print(self.sp_model.DecodeIds(cur_stat), end=" ")
                    print(self.sp_model.IdToPiece(x))
                    cur_stat = []
                else:
                    cur_stat.append(x)
            if cur_stat:
                print(self.sp_model.DecodeIds(cur_stat))

        data = pickle.load(open(filename, "rb"))
        convert_to_normal(data[200])

    def pad_data(self, data, max_len, reverse=False):
        """Pad the encoder input sequence with pad_id up to max_len."""
        pad_data = []
        for line in data:
            if len(line) >= max_len:
                if reverse:
                    line = line[-max_len:]
                else:
                    line = line[:max_len]
            else:
                while len(line) < max_len:
                    line.append(self.PAD)
            
            pad_data.append(line)
        return pad_data
    
    def pad_invoked_data(self, data, max_len, reverse=False):
        """Pad the encoder input sequence with pad_id up to max_len."""
        pad_data = []
        for line in data:
            if len(line) >= max_len:
                if reverse:
                    line = line[-max_len:]
                else:
                    line = line[:max_len]
            else:
                while len(line) < max_len:
                    line.append(0)
            
            pad_data.append(line)
        return pad_data

    def batch_iter(self, batch_size, state, epoch=None, shuffle=True, seed=12345):
        body_data = pickle.load(open(os.path.join(self.datapath, state+"_body.pkl"), "rb"))
        pro_data = pickle.load(open(os.path.join(self.datapath, state+"_pro.pkl"), "rb"))
        doc_data = pickle.load(open(os.path.join(self.datapath, state+"_doc.pkl"), "rb"))
        invoked_data = pickle.load(open(os.path.join(self.datapath, state+"_invoked.pkl"), "rb"))
        target_data = pickle.load(open(os.path.join(self.datapath, state+"_tag.pkl"), "rb"))

      
        print(len(target_data))
        assert len(body_data) == len(pro_data) == len(target_data)

        if shuffle:
            np.random.seed(seed)
            np.random.shuffle(body_data)
            np.random.seed(seed)
            np.random.shuffle(pro_data)
            np.random.seed(seed)
            np.random.shuffle(doc_data)
            np.random.seed(seed)
            np.random.shuffle(target_data)
            np.random.seed(seed)
            np.random.shuffle(invoked_data)
            
        dec_inp_data, dec_tgt_data = self.get_dec_inp_targ_seqs(target_data, self.tgt_name_len)

        body_data = self.pad_data(body_data, self.body_context_size)
        pro_data = self.pad_data(pro_data, self.project_context_size, True)
        doc_data = self.pad_data(pro_data, self.doc_context_size, True)
        invoked_data = self.pad_invoked_data(invoked_data, self.project_context_size, True)
        dec_inp_data = self.pad_data(dec_inp_data, self.tgt_name_len)
        dec_tgt_data = self.pad_data(dec_tgt_data, self.tgt_name_len)
        
        
        batch_len = int(len(target_data) // batch_size)
        print(batch_len)

        self.batch_len = batch_len

        all_body_data = body_data[:(batch_len * batch_size)]
        all_pro_data = pro_data[:(batch_len * batch_size)]
        all_doc_data = doc_data[:(batch_len * batch_size)]
        all_invoked_data = invoked_data[:(batch_len * batch_size)]
        all_target_data = target_data[:(batch_len * batch_size)]
        all_dec_inp_data = dec_inp_data[:(batch_len * batch_size)]
        all_dec_tgt_data = dec_tgt_data[:(batch_len * batch_size)]

        all_body_data = np.reshape(all_body_data, (batch_size, batch_len * self.body_context_size))
        all_pro_data = np.reshape(all_pro_data, (batch_size, batch_len * self.project_context_size))
        all_doc_data = np.reshape(all_doc_data, (batch_size, batch_len * self.doc_context_size))
        all_invoked_data = np.reshape(all_invoked_data, (batch_size, batch_len * self.project_context_size))
        
        all_dec_inp_data = np.reshape(all_dec_inp_data, (batch_size, batch_len * self.tgt_name_len))
        all_dec_tgt_data = np.reshape(all_dec_tgt_data, (batch_size, batch_len * self.tgt_name_len))

        for i in range(batch_len):
            # print('batch %d'%i)
            body_batch = all_body_data[:, i * self.body_context_size:(i + 1) * self.body_context_size]
            pro_batch = all_pro_data[:, i * self.project_context_size:(i + 1) * self.project_context_size]
            doc_batch = all_doc_data[:, i * self.doc_context_size:(i + 1) * self.doc_context_size]
            invoked_batch = all_invoked_data[:, i * self.project_context_size:(i + 1) * self.project_context_size]
            dec_inp_batch = all_dec_inp_data[:, i * self.tgt_name_len:(i + 1) * self.tgt_name_len]
            dec_tgt_batch = all_dec_tgt_data[:, i * self.tgt_name_len:(i + 1) * self.tgt_name_len]

            yield body_batch, pro_batch, doc_batch, dec_inp_batch, dec_tgt_batch, invoked_batch, batch_len


def parse_args():
    """
    Parse the args passed from the command line specifiying the specific conf.yaml to load
    """
    parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     "--vocab_file",
    #     type=str,
    #     default="bpe/java_spm.model",
    #     help="Path to sentencepiece vocabulary file",
    # )
    parser.add_argument(
        "--sub_vocab_file",
        type=str,
        default="/data4/liufang/GTNM/data_processing/sub_token_w2id.txt",
        help="Path to sub word vocabulary file",
    )
    parser.add_argument(
        "--doc_vocab_file",
        type=str,
        default="/data4/liufang/GTNM/data_processing/doc_w2id.txt",
        help="Path to documentation vocabulary file",
    )
    parser.add_argument(
        "--input_file_name",
        type=str, 
        default="/data4/liufang/GTNM/small-raw/java-small-train_all.pkl",
        help="Input file name",
    )
    parser.add_argument(
        "--output_file_name",
        type=str,
        default="/data4/liufang/GTNM/small-raw/saved/train",
        help="Output file name",
    )
    parser.add_argument(
        "--docstring",
        action="store_true",
        help="Whether to add docstring to context",
    )
   
    parser.add_argument(
        "--expr_max_len",
        type=int,
        default="1024",
        help="Max length of characters for a global assignment or a class attribute expression",
    )
    parser.add_argument(
        "--expr_max_num",
        type=int,
        default="30",
        help="Max number of a global assignment or a class attribute expression to keep",
    )
    return vars(parser.parse_args())


if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)-15s %(name)-5s %(levelname)-8s %(message)s",
    )
    LOGGER = logging.getLogger(__name__)

    args = parse_args()

    # vocab_file = args["vocab_file"]
    sub_vocab_file = args["sub_vocab_file"]
    doc_vocab_file = args["doc_vocab_file"]
    input_file_name = args["input_file_name"]
    output_file_name = args["output_file_name"]
    add_doc = args["docstring"]
    expr_max_len = args["expr_max_len"]
    expr_max_num = args["expr_max_num"]

    body_context_size = 55
    doc_context_size = 10
    project_context_size = 60
    tgt_name_size = 5
    main_patterns = ["if\s+__name__\s+==\s+'__main__':\n", 'if\s+__name__\s+==\s+"__main__":\n']
    # 1 represent the special token to seperate context and method body
    # assert (
    #     local_context_size + context_size == 1024
    # ), "The local and standard context should add up to 1024"

    processor = localContext(
        body_context_size, 
        doc_context_size,
        project_context_size,
        tgt_name_size,
        sub_vocab_file,  
        doc_vocab_file,
        include_docstring=add_doc, 
        expr_max_len=expr_max_len, 
        expr_max_num=expr_max_num,
        datapath = "/data4/liufang/GTNM/"
    )

    # processor.read_results(output_file_name)
    # exit()

    processor.extract_samples(input_file_name, output_file_name)
    
    # # processor.read_results(output_file_name[:-4]+"_body.pkl")
    # # print("====")
    # # processor.read_results(output_file_name[:-4]+"_pro.pkl")
    # # print("====")
    # # processor.read_results(output_file_name[:-4]+"_tag.pkl")
    # # print("====")

    # # data_loader = processor.batch_iter(8, "test")
    # # for body_batch, pro_batch, dec_inp_batch, dec_tgt_batch, batch_len in data_loader:
    # #     print(body_batch)
    # #     print(pro_batch)
    # #     print(dec_inp_batch)
    # #     print(dec_tgt_batch)
    # #     print('===========')

    #get_inproject_test_data('/data4/liufang/NewJavaMethodNameProcessing/')

    # w2id, id2w = update_vocab('sub_token_w2id.txt')
    # print(w2id["<endoftext>"])
