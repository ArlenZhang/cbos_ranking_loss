"""
    作者： 张龙印
    日期：2018年3月11日
    描述： 对数据信息的配置
"""
from code.utils.file_processor import *

class configuration:
    def __init__(self, load=False):
        self.word2ids_dict = self.ids2vec_dict = self.filename2ids_dict = self.wsj_word_list = None
        # 文件路径配置
        self.corpus_train_path = "../data/corpus/TRAINING"
        self.corpus_test_path = "../data/corpus/TEST"
        self.corpus_pdtb_path = "../../data/corpus/PDTB"
        self.connective_path = "../data/connective/connective.txt"
        self.connective_pkl_path = "../data/connective/connective.pkl"
        self.converted_corpus_train_path = "../data/converted_corpus/train"
        self.converted_corpus_test_path = "../data/converted_corpus/test1"
        self.segment_model_path = "../data/segmentation_model_pkl/model.pickle.gz"
        self.segment_vocab_path = "../data/segmentation_model_pkl/vocab.pickle.gz"

        # word2vec词向量库路径配置
        self.bc_path = "../data/word2vec/bc3200.pickle.gz"
        self.pdtb_bc_path = "../data/word2vec/pdtb_embedding.pkl"

        # 训练数据测试数据路径
        self.train_d_l_path = "../data/converted_corpus/train_d_l.pkl"
        self.test_d_l_path = "../data/converted_corpus/test_d_l.pkl"

        # word_ids字典文件路径
        self.wsj_word_list_path = "../data/wsj_word_list.pkl"
        self.word2ids_dict_path = "../data/pdtb_word2ids.pkl"
        self.ids2vec_dict_path = "../data/ids2vec.pkl"
        self.filename2ids_dict_path = "../data/filename2ids.pkl"

        # edu_ids字典文件路径
        self.word2ids_dict_path = "../data/edu2ids.pkl"
        self.eduids2vec_dict_path = "../data/ids2vec.pkl"
        # self.filename2ids_dict_path = "../data/filename2ids.pkl"

        # 无监督的句子表示部分的超参数配置
        self.w_r_lr = 0.0003
        self.w_r_window = 10  # 前后几个词作为context
        self.e_r_window = 10  # 前后几个句子作为context
        self.dim_word = 50  # 设置为50维的词向量表示
        self.dim_sentence = 100  # 假定用100维的向量表示句子

        # 开始配置
        if load:
            self.load_data()

    def load_data(self):
        # 加载word2ids, 从单词映射到单词在列表中的下标
        if os.path.exists(self.word2ids_dict_path):
            self.word2ids_dict = load_data(self.word2ids_dict_path)

        # 加载ids2vec，从ids映射到单词对应的词向量
        if os.path.exists(self.ids2vec_dict_path):
            self.ids2vec_dict = load_data(self.ids2vec_dict_path)

        # 加载从文件名到篇章的edu文本 以及 word_ids数组之间的映射
        if os.path.exists(self.filename2ids_dict_path):
            self.filename2ids_dict = load_data(self.filename2ids_dict_path)

        # 加载wsj词库
        if os.path.exists(self.wsj_word_list_path):
            self.wsj_word_list = load_data(self.wsj_word_list_path)

        # 加载edu2ids和eduids2


if __name__ == "__main__":
    co = configuration()
