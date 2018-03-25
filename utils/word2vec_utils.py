import random
import sys
from code.utils.corpus_util import *
import zipfile
import numpy as np
import tensorflow as tf
from code.utils.file_processor import *
from code.word2vec_model.model_config import *
sys.path.append('..')

def read_data(file_path):
    with zipfile.ZipFile(file_path) as f:
        words = tf.compat.as_str(f.read(f.namelist()[0])).split()
    return words

def build_vocab(words, visual_fld, word2ids_pkl):
    safe_mkdir(visual_fld)
    with open(os.path.join(visual_fld, 'vocab.tsv'), 'w') as file:
        dictionary = dict()
        index = 0
        dictionary["UNK"] = index
        file.write("UNK" + '\n')
        index += 1
        for word in words:
            if word not in dictionary.keys():
                dictionary[word] = index
                index += 1
                file.write(word + '\n')

        with open(word2ids_pkl, "wb") as f:
            pkl.dump(dictionary, f)
        index_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
        file.close()
    return dictionary, index_dictionary

def convert_words_to_index(words, dictionary):
    """
        Replace each word in the dataset with its index in the dictionary
    """
    return [dictionary[word] if word in dictionary else 0 for word in words]

def generate_sample(index_words, context_window_size):
    """
        Form training pairs according to the skip-gram model.
    """
    for index, center in enumerate(index_words):
        context = random.randint(1, context_window_size)
        # get a random target before the center word
        for target in index_words[max(0, index - context): index]:
            yield center, target
        # get a random target after the center wrod
        for target in index_words[index + 1: index + context + 1]:
            yield center, target

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

def batch_gen(batch_size, skip_window, use_edus=True):
    if CORPUS_TYPE == "Gigaword":
        if use_edus:
            words = get_text_by_edu(Giga_EDUS_PKL)  # 获取的从EDU文件得到的整体文本
        else:
            words = None
        # 创建词典
        dictionary, _ = build_vocab(words, VISUAL_FLD_Giga, Giga_WORD2IDS_PKL)
    else:
        if use_edus:
            words = get_text_by_edu(PDTB_EDUS_PKL)  # 默认选用PDTB
        else:
            words = None
        dictionary, _ = build_vocab(words, VISUAL_FLD_PDTB, PDTB_WORD2IDS_PKL)

    index_words = convert_words_to_index(words, dictionary)
    del words
    single_gen = generate_sample(index_words, skip_window)
    while True:
        center_batch = np.zeros(batch_size, dtype=np.int32)
        target_batch = np.zeros([batch_size, 1])
        for index in range(batch_size):
            center_batch[index], target_batch[index] = next(single_gen)
        yield center_batch, target_batch
