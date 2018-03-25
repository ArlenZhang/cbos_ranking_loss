"""
    作者：张龙印
    日期：2018.3.12
    为rst文件建立文档对象，根据edu文件位置建立一个doc对象

"""
import os
from code.utils.text_process_util import *

class rst_doc:
    def __init__(self, config_):
        self.config_ = config_
        self.filename_eduids_dict = dict()

    def create_edusids_dict(self):
        # 训练数据edu_ids生成
        TRAINING = self.config_.corpus_train_path
        for filename in os.listdir(TRAINING):
            temp_file = os.path.join(TRAINING, filename)
            temp_edusids_list = []
            with open(temp_file, "r") as f:
                for line in f:
                    edu_ids = self.get_ids_by_line(line)
                    temp_edusids_list.append(edu_ids)
            self.filename_eduids_dict[filename] = temp_edusids_list

        # 测试数据edu_ids生成
        TEST = self.config_.corpus_test_path
        for filename in os.listdir(TEST):
            temp_file = os.path.join(TEST, filename)
            temp_edusids_list = []
            with open(temp_file, "r") as f:
                for line in f:
                    edu_ids = self.get_ids_by_line(line)
                    temp_edusids_list.append(edu_ids)
            self.filename_eduids_dict[filename] = temp_edusids_list

    """
        根据line得到各个分词的在词库中的
    """
    def get_ids_by_line(self, line):
        ids = []
        word2ids_dict = self.config_.word2ids_dict
        line_list = line.split(" ")
        for item in line_list:
            item_ = rm_edge_s(item)
            if item_ in word2ids_dict.keys():
                ids.append(word2ids_dict[item_])
            else:
                ids.append(-1)
        return ids
