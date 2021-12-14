import argparse

class Hparams:
    parser = argparse.ArgumentParser()
    # prepro
    # parser.add_argument('--vocab_size', default=50000, type=int)
    parser.add_argument('--gpu', default='0', help='gpu id')

    parser.add_argument('--sub_word_vocab_file', default='/data4/liufang/GTNM/data_processing/sub_token_w2id.txt',
                        help="vocabulary file path")
    parser.add_argument('--doc_vocab_file', default='/data4/liufang/GTNM/data_processing/doc_w2id.txt',
                        help="vocabulary file path")
    
    parser.add_argument('--data_path', default='/data4/liufang/GTNM/',
                        help="data path")

    # training scheme
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--eval_batch_size', default=128, type=int)

    parser.add_argument('--lr', default=0.0001, type=float, help="learning rate")
    parser.add_argument('--warmup_steps', default=4000, type=int)
    parser.add_argument('--logdir', default="/data4/liufang/GTNM/saved", help="log directory")
    parser.add_argument('--num_epochs', default=50, type=int)
    parser.add_argument('--save_epochs', default=3, type=int)
    parser.add_argument('--evaldir', default="/data4/liufang/GTNM/", help="evaluation dir")

    parser.add_argument('--body_context_size', default=55, type=int,
                        help="body_context_size")
    parser.add_argument('--project_context_size', default=60, type=int,
                        help="project_context_size")
    parser.add_argument('--doc_context_size', default=10, type=int,
                        help="project_context_size")
    parser.add_argument('--tgt_name_size', default=5, type=int,
                        help="tgt_name_size")

    parser.add_argument('--pro', default=False, type=bool, 
                        help="whether to use project-specific info")

    # model
    parser.add_argument('--d_model', default=512, type=int,
                        help="hidden dimension of encoder/decoder")
    parser.add_argument('--d_ff', default=2048, type=int,
                        help="hidden dimension of feedforward layer")
    parser.add_argument('--num_blocks', default=6, type=int,
                        help="number of encoder/decoder blocks")
    parser.add_argument('--num_heads', default=8, type=int,
                        help="number of attention heads")
   
    parser.add_argument('--dropout_rate', default=0.3, type=float)
    parser.add_argument('--smoothing', default=0.1, type=float,
                        help="label smoothing rate")

    # test
    parser.add_argument('--ckpt', help="checkpoint file path")
    parser.add_argument('--test_batch_size', default=128, type=int)
    parser.add_argument('--testdir', default="", help="test result dir")
    parser.add_argument('--res_log', default="res.txt", help="result dir")
