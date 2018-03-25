# Model 超参数

CORPUS_TYPE = "pdtb"
if CORPUS_TYPE is "Gigaword":
    EDU_SIZE = 113693
else:
    EDU_SIZE = 113693
# 随机选择的篇章总数
RANDOM_DOC_NUM = 30000  # 选择5万个篇章进行训练

BATCH_SIZE = 1000
EMBED_SIZE = 128  # EDU表示的维度 和 word vector 表示中保持一致
SKIP_WINDOW = 2  # context上下文的单向跨度
NORM_SAMPLING_NUM = 30  # 从上下文按照正太分布抽取10句作为上下文环境
NUM_SAMPLED = 4  # 负采样个数
LEARNING_RATE = 0.003  # 学习率
NUM_TRAIN_STEPS = 100000  # 训练次数 100000
VISUAL_FLD_PDTB = 'visualization_e2v_pdtb'  # 对句子表示的展示
VISUAL_FLD_Gigaword = 'visualization_e2v_giga'  # 对句子表示的展示
SKIP_STEP = 5000  # 每5000次输出一次训练情况 打印loss等

# Parameters for downloading data
EXPECTED_BYTES = 31344016
NUM_VISUALIZE = 1000  # number of edus to visualize

# 针对pdtb语料库
PDTB_EDUS_PKL = "data/corpus/PDTB/edus.pkl"
PDTB_EDUS_FOLDER = "data/converted_corpus/pdtb"
PDTB_EDUS_2_IDS = "data/edu2vec/edu2ids_pdtb.pkl"
PDTB_IDS_2_VEC = "data/edu2vec/eduids2vec_pdtb.pkl"
PDTB_WORD_EMBEDDING = "data/word2vec/pdtb_embedding.pkl"

# 针对Giga语料库
Giga_EDUS_PKL = "data/corpus/Gigaword/edus.pkl"
Giga_EDUS_FOLDER = "data/converted_corpus/giga"
Giga_EDUS_2_IDS = "data/edu2vec/edu2ids_giga.pkl"
Giga_IDS_2_VEC = "data/edu2vec/eduids2vec_giga.pkl"
Giga_WORD_EMBEDDING = "data/word2vec/giga_embedding.pkl"

# 最终的到
PDTB_EDU_EMBEDDING_PKL = "data/edu2vec/edu_embedding_pdtb.pkl"
Giga_EDU_EMBEDDING_PKL = "data/edu2vec/edu_embedding_giga.pkl"

PDTB_EDU_TSV = "visualization_e2v_pdtb/vocab.tsv"

# 避免上溢出，添加调节参数
AVOID_MAX = 80.0
