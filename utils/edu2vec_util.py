"""
    进行PDTB中EDU分子句，完成edu2vec模型的一些数据处理
"""
from code.utils.text_process_util import *
from code.utils.file_processor import *
import numpy as np
import os
from code.edu2vec_model.model_config import *

def most_common_words(visual_fld, num_visualize):
    """
        create a list of num_visualize most frequent words to visualize on TensorBoard.
        saved to visualization/vocab_[num_visualize].tsv
    """
    words = open(os.path.join(visual_fld, 'vocab.tsv'), 'r').readlines()[:num_visualize]
    words = [word for word in words]
    file = open(os.path.join(visual_fld, 'vocab_' + str(num_visualize) + '.tsv'), 'w')
    for word in words:
        file.write(word)
    file.close()

def load_edus(path):
    if os.path.exists(path):
        edus_list = load_data(path)
    else:
        # 从切割的EDU构建最新pkl文件
        edus_list = []
        for filename in os.listdir(PDTB_EDUS_FOLDER):
            if filename.endswith(".edus"):
                temp_f = os.path.join(PDTB_EDUS_FOLDER, filename)
                with open(temp_f, "r") as f:
                    for line in f:
                        edus_list.append(line)
        save_data(edus_list, path)
    return edus_list

"""
    根据 edus 创建edu到ids之间的映射
    传入的是segmentation生成的EDUs文件
    对创建好的edu2vec得到的edu向量进行归一化处理
"""
# 生成并存储edu2ids文件, 根据word_embedding计算初始化的edu embedding
def build_edu_lib(edus):
    # 加载word_embedding
    if CORPUS_TYPE is "Gigaword":
        word_embedding = load_data(Giga_WORD_EMBEDDING)
        safe_mkdir(VISUAL_FLD_Gigaword)
        tmp_path = VISUAL_FLD_Gigaword
        save_edu2ids = Giga_EDUS_2_IDS
        save_ids2vec = Giga_IDS_2_VEC
    else:
        word_embedding = load_data(PDTB_WORD_EMBEDDING)
        safe_mkdir(VISUAL_FLD_PDTB)
        tmp_path = VISUAL_FLD_PDTB
        save_edu2ids = PDTB_EDUS_2_IDS
        save_ids2vec = PDTB_IDS_2_VEC

    with open(os.path.join(tmp_path, 'vocab.tsv'), 'w') as file:
        edu2ids = dict()
        index = 1  # edu从1开始编号，第0位置设置为填补空位的向量，设置为全0
        eduids2vec = np.zeros(shape=(1, EMBED_SIZE))  # 对0位置填补0向量
        file.write("UNK" + '\n')
        for edu in edus:
            if edu not in edu2ids.keys():
                # 计算edu当前的embedding根据之前训练的word vector得到初始化的edu_embedding
                temp_word_list = edu.strip().lower().split()
                temp_edu_vec = None
                for word in temp_word_list:
                    # 对word净化处理
                    word_ = rm_edge_s(word)
                    if len(word_) == 0:
                        continue
                    # 查找word对应的向量表示
                    if word_ not in word_embedding.keys():
                        input("请查看word2vec词库，词库中缺失")
                        continue
                    word_vec = np.array(word_embedding[word_], dtype=np.float32)
                    if temp_edu_vec is None:
                        temp_edu_vec = word_vec
                    else:
                        temp_edu_vec = np.add(temp_edu_vec, word_vec)
                # 不加入没有能量的EDU
                if temp_edu_vec is None:
                    continue
                else:
                    # 当前EDU有能量，则加入映射
                    edu2ids[edu] = index
                    # 当前EDU有能量temp_edu_vec不为空，则当前向量值
                    temp_edu_vec = np.array([temp_edu_vec])
                    if eduids2vec is None:
                        eduids2vec = temp_edu_vec
                    else:
                        # 进行拼接
                        eduids2vec = np.append(eduids2vec, temp_edu_vec, axis=0)  # 按照列进行拼接
                    # 写入文件
                    file.write(edu.strip() + '\n')
                    index += 1
    # 数据归一化
    eduids2vec = normalize(eduids2vec)
    # 将固定数据存储
    save_data(edu2ids, save_edu2ids)
    save_data(eduids2vec, save_ids2vec)
    file.close()

def normalize(eduids2vec):
    min_ = np.min(eduids2vec)
    max_ = np.max(eduids2vec)
    eduids2vec = np.divide(np.subtract(eduids2vec, min_), max_ - min_)
    return eduids2vec

def convert_edus2ids(edus, edu2ids):
    edu_ids_ = []
    for edu in edus:
        if edu in edu2ids.keys():
            edu_ids_.append(edu2ids[edu])
    return np.array(edu_ids_)

"""
    根据edu的正太分布获取上下文
    
    # mu设置为当前中心  
    mu, sigma = 0.5, .1
    s = np.random.normal(loc=mu, scale=sigma, size=10)
    
    # 整个一批edu个数 batch_size
    sample_id = int(batch_size * s)
    
    print(s)
    
    BATCH_SIZE的设置和平均每篇文章的edu个数有关
"""
def generate_sample_norm(index_edus):
    up_boundary = len(index_edus) - 1
    # 对处理得到的EDU下标列表进行处理
    for index, edu_id in enumerate(index_edus):
        context_ids = np.zeros(shape=(NORM_SAMPLING_NUM,), dtype=np.int32)
        temp_edu_p, sigma = index/BATCH_SIZE, 1
        s = np.random.normal(loc=temp_edu_p, scale=sigma, size=NORM_SAMPLING_NUM)  # 以当前edu_id在当前批次中的百分比确定

        context_ids[idx] = 0

        # 获取负采样结果, 取指定个数的随机的edu作为负样本，之后可能会做更多技巧
        # 注意这里选用的是EDU编号，抛出0,最大值范围根据字典中edu个数确定, 不包括最大值，注意点
        target_ids = np.random.randint(1, EDU_SIZE, NUM_SAMPLED + 1)
        target_ids[-1] = edu_id  # 正样本

        # 标志位
        target_ids_tag = np.zeros(NUM_SAMPLED + 1)
        target_ids_tag[-1] = 1  # 标志 正例

        yield context_ids, target_ids, target_ids_tag

"""
    可迭代地生成一批数据中的单个元素
    返回值：
        当前EDU上下文环境的edu下标
        当前EDU下，取的负样本向量值列表下标
"""
def generate_sample(index_edus):
    up_boundary = len(index_edus)-1
    # 对处理得到的EDU下标列表进行处理
    for index, edu_id in enumerate(index_edus):
        # 构建上下文环境 上边界是 i+k 下边界是i-k 长度是2*k + 1
        # context = random.randint(1, SKIP_WINDOW)
        context_ids = np.zeros(shape=(2*SKIP_WINDOW+1,), dtype=np.int32)
        idx = 0
        # 如果出现最首部 idx 会累加，否则不会
        while index - SKIP_WINDOW + idx < 0:
            context_ids[idx] = 0  # edi_ids中0位置放置的是全0向量
            idx += 1
        lower_idx = index - SKIP_WINDOW + idx

        for edus_ in index_edus[lower_idx: min(up_boundary + 1, index + SKIP_WINDOW + 1)]:
            context_ids[idx] = edus_
            idx += 1

        # 控制高位
        while idx < 2 * SKIP_WINDOW + 1:
            context_ids[idx] = 0
            idx += 1
        # 获取负采样结果, 取指定个数的随机的edu作为负样本，之后可能会做更多技巧
        # 注意这里选用的是EDU编号，抛出0,最大值范围根据字典中edu个数确定, 不包括最大值，注意点
        target_ids = np.random.randint(1, EDU_SIZE, NUM_SAMPLED + 1)
        target_ids[-1] = edu_id  # 正样本

        # 标志位
        target_ids_tag = np.zeros(NUM_SAMPLED+1)
        target_ids_tag[-1] = 1  # 标志 正例

        yield context_ids, target_ids, target_ids_tag

def load_edu_embedding():
    if CORPUS_TYPE is "Gigaword":
        # 获取训练数据
        edus = load_edus(Giga_EDUS_PKL)
        # 检测是否加载好训练数据的edu_ids 没有就添加
        try:
            os.stat(Giga_EDUS_2_IDS)
            os.stat(Giga_IDS_2_VEC)
        except OSError:
            build_edu_lib(edus)
        edu_embedding = load_data(Giga_IDS_2_VEC)

    else:
        edus = load_edus(PDTB_EDUS_PKL)
        try:
            os.stat(PDTB_EDUS_2_IDS)
            os.stat(PDTB_IDS_2_VEC)
        except OSError:
            build_edu_lib(edus)
        edu_embedding = load_data(PDTB_IDS_2_VEC)
    return edu_embedding

# 根据生成好的EDU文件获取edus
def batch_gen():
    if CORPUS_TYPE is "Gigaword":
        # 获取训练数据
        edus = load_edus(Giga_EDUS_PKL)
        # 加载edu2ids 和 ids2vec
        edu2ids, _ = load_data(Giga_EDUS_2_IDS), load_data(Giga_IDS_2_VEC)
        edu_ids = convert_edus2ids(edus, edu2ids)  # 最终得到的训练数据
    else:
        # 获取训练数据
        edus = load_edus(PDTB_EDUS_PKL)
        # 加载edu2ids 和 ids2vec
        edu2ids, _ = load_data(PDTB_EDUS_2_IDS), load_data(PDTB_IDS_2_VEC)
        edu_ids = convert_edus2ids(edus, edu2ids)  # 最终得到的训练数据
    # 构建迭代器
    # single_gen = generate_sample(edu_ids)
    # 根据正太分布进行上下文环境获取版本
    single_gen = generate_sample_norm(edu_ids)

    # context_batch 存储上下文EDU下标,  target_sent 存储整例和负例EDU下标
    while True:
        # target_samples_tags = np.zeros(NUM_SAMPLED+1, dtype=np.int32)
        context_batch = np.zeros([BATCH_SIZE, 2*SKIP_WINDOW+1], dtype=np.int32)  # 每一行存储上下文环境EDU对应下标
        context_batch_norm = np.zeros([BATCH_SIZE, NORM_SAMPLING_NUM], dtype=np.int32)  # 正太分布的上下文抽取
        target_batch = np.zeros([BATCH_SIZE, NUM_SAMPLED+1], dtype=np.int32)  # 存储负采样和正采样
        target_tag_batch = np.zeros([BATCH_SIZE, NUM_SAMPLED+1], dtype=np.float32)
        for index in range(BATCH_SIZE):
            context_batch[index], target_batch[index], target_tag_batch[index] = next(single_gen)
        yield context_batch, target_batch, target_tag_batch

if __name__ == "__main__":
    _, ids2vec = load_data(PDTB_EDUS_2_IDS), load_data(PDTB_IDS_2_VEC)
    print(ids2vec[0])
