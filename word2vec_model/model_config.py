# Model 超参数
CORPUS_TYPE = "pdtb"  # 或者pdtb

if CORPUS_TYPE is "Gigaword":
    VOCAB_SIZE = 41392  # 加一个UNK
else:
    VOCAB_SIZE = 41392  # 加上一个UNK

BATCH_SIZE = 100
EMBED_SIZE = 128  # dimension of the word embedding vectors 和 edu表示中保持一致
SKIP_WINDOW = 1  # the context window
NUM_SAMPLED = 64  # number of negative examples to sample
LEARNING_RATE = 0.7
NUM_TRAIN_STEPS = 100000  # 训练100000次
VISUAL_FLD_PDTB = 'visualization_w2v_pdtb'
VISUAL_FLD_Giga = 'visualization_w2v_giga'
SKIP_STEP = 5000  # 每5000次输出一次训练情况 打印loss等

# Parameters for downloading data
DOWNLOAD_URL = 'http://mattmahoney.net/dc/text8.zip'
EXPECTED_BYTES = 31344016
NUM_VISUALIZE = 100  # number of tokens to visualize

# segmentation部分
SEGMENT_MODEL_PATH = "data/segmentation_model_pkl/model.pickle.gz"
SEGMENT_VOCAB_PATH = "data/segmentation_model_pkl/vocab.pickle.gz"
CONVERTED_CORPUS_PDTB_PATH = "data/converted_corpus/pdtb"

# PDTB
PDTB_WORD2IDS_PKL = "data/word2vec/pdtb_word2ids.pkl"
PDTB_EMBEDDING_PKL = "data/word2vec/pdtb_embedding.pkl"

PDTB_EDUS_FOLDER = "data/converted_corpus/pdtb"
PDTB_EDUS_PKL = "data/corpus/PDTB/edus.pkl"
PDTB_CORPUS_EDUS_DICT = "data/corpus/PDTB/corpus_edus_dict.pkl"

# Gigaword
Giga_WORD2IDS_PKL = "data/word2vec/giga_word2ids.pkl"
Giga_EMBEDDING_PKL = "data/word2vec/giga_embedding.pkl"

Giga_EDUS_FOLDER = "data/converted_corpus/giga"
Giga_EDUS_PKL = "data/corpus/Gigaword/edus.pkl"
Giga_CORPUS_EDUS_DICT = "data/corpus/Gigaword/corpus_edus_dict.pkl"

Giga_Corpus_RAWb = "data/corpus/Gigaword/raw_bin"
Giga_Corpus_Raw = "data/corpus/Gigaword/raw"
