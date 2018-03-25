"""
    整个处理过程 只负责RST部分，其余见各自model
    作者：张龙印
    日期：2018.3.11
"""
from code.config import configuration
from code.segmenter import *
from code.utils.data_util import *
from code.utils.segment_util import *

def convert_data(config_):
    conn_pkl_convert(config_)
    # 如果converted中的xml文件已经准备好, 构建conll文件
    extract(config_.converted_corpus_train_path)
    extract(config_.converted_corpus_test_path)
    create_word2ids2vec_dict(config_)
    create_filename2ids_dict(config_)

def do_segmentation(config_):

    # 生成merge数据
    seg(config_.segment_model_path, config_.segment_vocab_path, config_.converted_corpus_train_path)  # train
    seg(config_.segment_model_path, config_.segment_vocab_path, config_.converted_corpus_train_path)  # test1
    # 根据得到的merge文件构建edus文件
    push_edus_file(config_.converted_corpus_train_path)
    push_edus_file(config_.converted_corpus_test_path)

def prepare_train_data_(config_):
    ptd = PTD(config_=config_)
    ptd.prep_train_data()


if __name__ == "__main__":
    # 配置信息准备
    config = configuration(load=True)
    # convert_data(config)
    # do_segmentation(config)
    # 根据标准数据和shift reduce过程 生成训练数据
    prepare_train_data_(config)
