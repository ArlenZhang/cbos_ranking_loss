"""
    pdtb部分数据处理
"""
from code.word2vec_model.model_config import *
from code.edu2vec_model.model_config import RANDOM_DOC_NUM
from code.utils.segment_util import *
from code.segmenter import *
from code.utils.file_processor import *
import nltk
import numpy as np
import time
import nltk.data
# 定义取文本的reg
flag = False

# 定义每个文件中的篇章id，每个文件存储若干个子文件作为篇章文件，命名方式根据文件名_docID作为篇章的名，每次解析放进一个篇章而不能是很多很多篇章
doc_id = 0

# 定义一个全局文件名用于巨鹿最新篇章名称
doc_file_name = ""

# 定义当前要存出的篇章内容
file_content = ""

ave_rand_doc_num_item = 0

tmp_rand_flag = 0

is_end = False

def segmentation_(converted_corpus):
    if os.path.exists(os.path.join(converted_corpus, "ok")):
        print("分子句工作已经做过！")
        return
    extract(converted_corpus)  # 对xml文件的提取
    # 生成merge数据
    seg(SEGMENT_MODEL_PATH, SEGMENT_VOCAB_PATH, converted_corpus)
    push_edus_file(converted_corpus)
    with open(os.path.join(converted_corpus, "ok"), 'w') as f:
        f.write("转换时间：" + str(time.strftime('%Y-%m-%d', time.localtime(time.time()))))

def create_one_edu2list_dict(edus_folder, edus_list_pkl, edus_dict_pkl):
    # 创建edu pkl文件
    edus_list = []
    corpus_dict = dict()
    for filename in os.listdir(edus_folder):
        if filename.endswith(".edus"):
            context_list = []
            temp_f = os.path.join(edus_folder, filename)
            with open(temp_f, "r") as f:
                for line in f:
                    line = line.strip()
                    edus_list.append(line)
                    context_list.append(line)
            corpus_dict[filename] = context_list

    save_data(edus_list, edus_list_pkl)
    save_data(corpus_dict, edus_dict_pkl)

"""
    从来源列表中对应的文件区域随机抽取 RANDOM_DOC_NUM 个篇章作为语料库
    输入：
        raw_bin: 分来源的压缩文件
        type_: 来源列表，选择与RST关联大一点的数据
"""
def giga_convert(raw_bin, type_):
    ave_rand_doc_num = RANDOM_DOC_NUM/float(len(type_))
    print("每个来源的选择数目： ", ave_rand_doc_num)
    input("总的来源个数：" + str(len(type_)))
    global doc_id, doc_file_name, file_content, ave_rand_doc_num_item, is_end
    for filename in os.listdir(raw_bin):
        if filename not in type_:
            continue
        doc_file_name = filename
        input("当前随机进入的文件： " + filename)
        temp_dir = os.path.join(raw_bin, filename)
        chile_dir_list = os.listdir(temp_dir)
        input("当前来源的子文件夹个数："+str(len(chile_dir_list)))
        ave_rand_doc_num_item = ave_rand_doc_num / len(chile_dir_list)  # 计算每个篇章文件中抽取的文章数量
        input("每个子文件夹中随机数目："+str(ave_rand_doc_num_item))
        for itemname in chile_dir_list:
            # doc_file_name = itemname[:-3]  # 每读到一个新的文件就更新篇章文件名
            # 创建属类下面的文件名的文件夹
            # 下面对文件中的若干篇章进行提取
            temp_file = os.path.join(temp_dir, itemname)
            g_file = gzip.GzipFile(temp_file)
            for line in g_file:
                tmp_line = reg_text(line.decode("utf-8"))
                if is_end:
                    is_end = False
                    break
                if not flag:  # 对空行或者没有问本的行的越过
                    continue
                # 上面是True也就是<p>开始，如果随机数为1，则加入文本数据
                file_content += tmp_line + " "

# 对每一行进行文本提取
def reg_text(txt):
    global flag, doc_id, file_content, tmp_rand_flag, is_end
    if "<DOC id=" in txt:
        tmp_rand_flag = np.random.randint(0, 2)  # 选择0或者1
        doc_id += tmp_rand_flag  # 每次遇到新的篇章就累加1,因此文件名从1开始
        file_content = ""  # 每次遇到新的篇章就初始化要存储的内容为空
        return ""
    if "</DOC>" in txt:
        if tmp_rand_flag == 1:
            tmp_rand_flag = 0
            # 遇到篇章结束，直接开始写文件
            save_doc()
        # 判断是否全部选择完毕
        if doc_id >= ave_rand_doc_num_item:
            doc_id = 0
            # 结束当前文件的阅读
            is_end = True
        return ""
    if "<P>" in txt:
        flag = True
        return ""
    if "</P>" in txt:
        flag = False
        return ""
    return txt.strip()

# 根据全局变量，当前读取数据的篇章文件名和当前的篇章id获取名字存储当前数据
def save_doc():
    global doc_id
    if len(file_content.strip()) == 0:  # 如果当前篇章无内容，自动忽略
        doc_id -= 1
    else:
        with open(os.path.join(Giga_Corpus_Raw, doc_file_name+str(doc_id)), "w") as f:
            # 将文件分句
            write_file_content = split_sentence(file_content)
            f.write(write_file_content)
            print("生成文件：", doc_file_name, str(doc_id))

def split_sentence(paragraph):
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    sentences = tokenizer.tokenize(paragraph)
    return '\n'.join(sentences)
