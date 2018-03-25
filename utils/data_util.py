"""
    对数据进行处理，为了条理，统一放那盖在此处进行数据处理
    作者：张龙印
    日期：2018.3.11
"""
import _pickle as cPickle
from code.utils.file_processor import *
from code.data_struct.rst_tree import rst_tree
from code.data_struct.rst_tree import edus_list
from code.utils.text_process_util import *

def conn_pkl_convert(config_):
    f_obj = open(config_.connective_path, "r")
    word_set = set()
    leng = 0
    flag = " "
    for line in f_obj:
        if len(line.split(" ")) > 3:
            continue
        line = line.strip()
        line = line.lower()
        if line not in word_set:
            word_set.add(line)
            if len(line) > leng:
                leng = len(line)
                flag = line
    save_data(word_set, config_.connective_pkl_path)
    print("longest: ", leng, " flag: ", flag)

"""
    根据生成的词库构建一个词库到ids的字典
"""
def create_word2ids2vec_dict(config_, type_="bc"):
    if not os.path.exists(config_.word2ids_dict_path):
        word2ids_dict = dict()
        ids2vec_dict = dict()
        if type_ == "bc":
            with gzip.open(config_.bc_path, "rb") as f:
                bc_dict = pkl.load(f)
        else:
            with gzip.open(config_.pdtb_bc_path, "rb") as f:
                bc_dict = pkl.load(f)
        idx = 0
        word_list = create_word_list(config_, bc_dict)
        for word in word_list:
            word2ids_dict[word] = idx
            ids2vec_dict[idx] = bc_dict[word]
            idx += 1
        # 存储
        save_data(word2ids_dict, config_.word2ids_dict_path)
        save_data(ids2vec_dict, config_.ids2vec_dict_path)

"""
    创建词库，匹配bc中的词汇和语料库中的词汇生成小的词库
    怎么训练一个针对RST的word2vec出来，加载这个embedding之后bc_words中就存在这些训练词汇
    共171399词，loss 8178个, 380篇wsj    
"""
def create_word_list(config_, bc_dict):
    word_list = []
    wsj_word_set = set()
    bc_words = bc_dict.keys()
    for filename in os.listdir(config_.corpus_train_path):
        if filename.endswith(".out.edus"):
            file_path = os.path.join(config_.corpus_train_path, filename)
            with open(file_path, "r") as f:
                for line in f:
                    words = line.strip().lower().split(" ")
                    for item in words:
                        wsj_word_set.add(item)
                        print("pass_train: ", item)
                        item_ = rm_edge_s(item)
                        if item_ in bc_words:
                            word_list.append(item_)
                        else:
                            pass
    for filename in os.listdir(config_.corpus_test_path):
        if filename.endswith(".out.edus"):
            file_path = os.path.join(config_.corpus_test_path, filename)
            with open(file_path, "r") as f:
                for line in f:
                    words = line.strip().lower().split(" ")
                    for item in words:
                        wsj_word_set.add(item)
                        print("pass_test: ", item)
                        item_ = rm_edge_s(item)
                        if item_ in bc_words:
                            word_list.append(item_)
                        else:
                            pass
    # 将wsj中的所有词汇存储
    wsj_word_list = list(wsj_word_set)
    save_data(wsj_word_list, config_.wsj_word_list_path)
    return word_list


"""
    封装 文件名到edu list 的映射，文件名到edu_word_ids的映射
    filename2ids_dict 存储了ids和edu本身
"""
def create_filename2ids_dict(config_):
    if not os.path.exists(config_.filename2ids_dict_path):
        filename2ids_dict = dict()
        # 训练数据edu_ids生成
        TRAINING = config_.corpus_train_path
        for filename in os.listdir(TRAINING):
            if filename.endswith(".out.edus"):
                ids_key = str(filename.split(".")[0]) + "_ids"
                edu_key = str(filename.split(".")[0]) + "_edus"
                temp_file = os.path.join(TRAINING, filename)
                temp_edusids_list = []
                temp_edu_list = []
                with open(temp_file, "r") as f:
                    for line in f:
                        edu_ids, new_line = get_ids_by_line(config_, line)
                        temp_edusids_list.append(edu_ids)
                        temp_edu_list.append(new_line)
                filename2ids_dict[ids_key] = temp_edusids_list
                filename2ids_dict[edu_key] = temp_edu_list

        # 测试数据edu_ids生成
        TEST = config_.corpus_test_path
        for filename in os.listdir(TEST):
            if filename.endswith(".out.edus"):
                ids_key = str(filename.split(".")[0]) + "_ids"
                edu_key = str(filename.split(".")[0]) + "_edus"
                temp_file = os.path.join(TEST, filename)
                temp_edusids_list = []
                temp_edu_list = []
                with open(temp_file, "r") as f:
                    for line in f:
                        edu_ids, new_line = get_ids_by_line(config_, line)
                        temp_edusids_list.append(edu_ids)
                        temp_edu_list.append(new_line)

                filename2ids_dict[ids_key] = temp_edusids_list
                filename2ids_dict[edu_key] = temp_edu_list
        save_data(filename2ids_dict, config_.filename2ids_dict_path)

"""
    根据line得到各个分词的在词库中的
"""
def get_ids_by_line(config_, line):
    ids = []
    word2ids_dict = config_.word2ids_dict
    new_line = line.strip().lower()
    line_list = new_line.split(" ")
    for item in line_list:
        item_ = rm_edge_s(item)
        if item_ in word2ids_dict.keys():
            ids.append(word2ids_dict[item_])
        else:
            ids.append(-1)
    return ids, new_line


class PTD:
    def __init__(self, config_):
        self.config_ = config_
        # self.path=path
        # 创建字典，c对应操作
        self.train_data = dict()
        self.edu_stack = []
        self.edu_queue = []

        # 生成的训练数据
        self.train_data = []
        self.train_label = []

        # 生成测试数据
        self.test_data = []
        self.test_label = []

        # 当前文本
        self.doc = None
        # Brown Cluster
        with gzip.open(config_.bc_path) as fin:
            self.bcvocab = cPickle.load(fin)

    # 生成训练数据
    def prep_train_data(self):
        dis_file_dir = self.config_.corpus_train_path
        # merge_file_dir = config_.converted_corpus_train_path
        for file_name in os.listdir(dis_file_dir):
            if file_name.endswith('0600.out.dis'):
                temp_path = os.path.join(dis_file_dir, file_name)
                dis_tree_obj = open(temp_path, 'r')
                root = rst_tree(type_="null", lines_list=dis_tree_obj.readlines(), p_node=None)
                root.create_tree(temp_line_num=0, p_node_=root)
                # 初始化queue stack
                self.init_queue_stack(tree_root=root, file_name=file_name)
                root.pre_traverse()
                # 后续遍历生成训练数据
                self.traverse_data_label(root)

        # 遍历完所有文件和组态与操作之后，存储数据
        save_data((self.train_data, self.train_label), self.config_.train_d_l_path)
        save_data((self.test_data, self.test_label), self.config_.test_d_l_path)

    """
        作者：张龙印
        日期：2018.3.12
        描述：初始化队列和栈，栈空，队列满。输入一个建立好的篇章树，将节点对象入队列, 前序遍历将叶子节点存放入队列
    """

    def init_queue_stack(self, tree_root, file_name):
        self.edu_stack = []
        # 首先根据当前文件名创建当前文件的edu和ids数据
        edu_key = str(file_name.split(".")[0]) + "_edus"
        ids_key = str(file_name.split(".")[0]) + "_ids"
        tmp_edu_file = self.config_.filename2ids_dict[edu_key]
        tmp_ids_file = self.config_.filename2ids_dict[ids_key]
        tree_root.pre_traverse_leaves(temp_node=tree_root, temp_edu_file=tmp_edu_file, temp_eduids_file=tmp_ids_file)
        # 在新文件的时候对文件树前根遍历得到edu节点作为队列初始化数据
        self.edu_queue = edus_list
        print("queue number: ", len(self.edu_queue))
        print("stack number: ", len(self.edu_stack))

    """
        作者：张龙印
        日期：2018.3.12
        描述：当前queue中存放了都是叶子节点，后根遍历，shift过程直接从queue中取，合并过程直接从树节点给
    """

    def traverse_data_label(self, root):
        # 对遍历的每个状态得到的stack and queue进行特征获取 doc对象传过来
        if root is None:
            return
        else:
            self.traverse_data_label(root.left_child)
            self.traverse_data_label(root.right_child)
            # 对当前两个节点之间的关系，以及当前的组态configuration 进行存储
            feature = self.fetch_fe()

            # 取当前操作,只要当前结点没有孩子节点就对当前节点进行shift, 只要当前节点存在左右孩子就对左右孩子进行reduce,取左右孩子的rel
            if root.left_child is None and root.right_child is None:
                # 根据 gold_tree 构建 spanNode
                temp = self.edu_queue.pop(0)
                self.edu_stack.append(temp)
                temp_transition = "Shift"
            else:
                # 根据gold tree 提取下面数据
                child_NS_rel = root.child_NS_rel
                child_rel = root.child_rel
                # 顶层节点出栈
                self.edu_stack.pop()
                self.edu_stack.pop()
                # 把前两个节点删除之后，把新的节点加入到stack
                self.edu_stack.append(root)
                temp_transition = "Reduce-" + child_NS_rel + "-" + child_rel
            print("-------------------------------------------------------------------------")
            print("queue number: ", len(self.edu_queue))
            print("stack number: ", len(self.edu_stack))
            # 对训练数据进行存储
            self.train_data.append(feature)
            # 对当前数据和操作标签进行存储
            self.train_label.append(temp_transition)
            print("train_data len: ", len(self.train_data))
            print("train_label len: ", self.train_label)

    '''
        动态初始化，根据遍历到的文档初始化队列，栈设置为空
    '''

    def fetch_fe(self):
        return None
