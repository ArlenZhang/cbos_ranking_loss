from code.utils.text_process_util import *
from code.utils.file_processor import *
from code.segmenter import *

def get_pdtb_text_by_edu(pdtb_edus_pkl):
    edus_list = get_pdtb_edus(pdtb_edus_pkl)
    # 拼接文本数据
    result_text = ""
    for line in edus_list:
        result_text += line.strip() + " "
    # 全部转换成小写的英文
    result_words = result_text.lower().split()
    # 边缘处理
    word_list = list()
    # word_set = set()   # 统计词汇量
    for word in result_words:
        word_ = rm_edge_s(word)
        if len(word_) == 0:
            continue
        word_list.append(word_)
        # word_set.add(word_)
    return word_list  # 获取原始语料，文本整体获取。每一行作为一句的形式,返回句子列表,一个篇章一个list，一句话作为一个

# 获取edus列表
def get_pdtb_edus(pdtb_edus_pkl):
    if os.path.exists(pdtb_edus_pkl):
        # 直接加载数据
        edus_list = load_data(pdtb_edus_pkl)
    else:
        edus_list = None
        input("情先执行prepare_data.py生成每个语料的edu信息")
    return edus_list
