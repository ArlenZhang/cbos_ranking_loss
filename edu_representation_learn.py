from code.word2vec_model.prepare_data import *
from code.word2vec_model.model_config import *
from code.word2vec_model.word2vec import main_word
from code.edu2vec_model.cbos_model import main_edu

# 调用，对指定语料库进行分子句，构建映射
def prepare_dt(type_=CORPUS_TYPE):
    if type_ is "Gigaword":
        # 在处理giga_word文件之前，将四个文件存储到同一个文件，即将Giga_EDUS_FOLDER下面的四分文件中的所有文件转移到raw下面
        # type_ = ["afp_eng", "apw_eng", "cna_eng", "ltw_eng", "nyt_eng", "xin_eng"]
        type_ = ["nyt_eng", "xin_eng"]
        # type_ = ["test1", "test2"]
        giga_convert(Giga_Corpus_RAWb, type_)  # 根据指定参数从语料库中随机抽取指定量的数据到raw下，重新编排顺序
        input("请调用脚本完成out文件构建!")
        segmentation_(Giga_EDUS_FOLDER)
        create_one_edu2list_dict(Giga_EDUS_FOLDER, Giga_EDUS_PKL, Giga_CORPUS_EDUS_DICT)
    else:
        # 对几个语料库进行分子句，创建edu文件
        segmentation_(PDTB_EDUS_FOLDER)
        # 创建EDU列表文件  创建 edu_dict，从filename到文件edu_list之间的映射, 多语料都可以
        create_one_edu2list_dict(PDTB_EDUS_FOLDER, PDTB_EDUS_PKL, PDTB_CORPUS_EDUS_DICT)

def train_word_model():
    main_word()

def train_edu_model():
    main_edu()

if __name__ == "__main__":
    prepare_dt(type_="pdtb")
    train_word_model()
    train_edu_model()
