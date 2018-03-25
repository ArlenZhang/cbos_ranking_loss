import _pickle as cPickle
import gzip
import os
import pickle as pkl

# from code.datastructure import *
# from code.docreader import DocReader
# from code.feature import FeatureGenerator

from code.data_struct.rst_tree import rst_tree
# from code.model import parsing_model
# from code.vocab import vocab_create

"""
    语料需要转换，转换前对
    需要结合fetures特征,所以能够根据当前树的状态，得到对应的doc对象才能获取对应的特征
    遍历过程中
    
    验证树建立的合法性，进行前序遍历
            1希望建立的树和文件保持一致
            2希望shift-reduce过程和configuration保持同步，shift-reduce过程和文件的
        后续遍历保持一致，验证。
"""
class prepare_train_data:
    def __init__(self):
        # self.path=path
        # 创建字典，c对应操作
        self.train_data = dict()
        self.edu_stack = []
        self.edu_queue = []
        # 根据当前doc的merge文件对队列进行初始化
        # self.dr = DocReader()
        self.rst_tree = rst_tree("obj")
        # 训练数据对应的特征采集
        self.sample_list = []
        # 生成的训练数据
        self.train_data = []
        # 生成的训练标签
        self.train_label = []
        # 当前文本
        self.doc = None
        # Brown Cluster
        with gzip.open("resources/bc3200.pickle.gz") as fin:
            self.bcvocab = cPickle.load(fin)

        # 测试变量
        self.count = 1

        # 统计
        self.total_file_num = 0
        self.total_edu_num = 0
    '''
        后序遍历,进行序列提取(将当前stack和queue作为参数传递给函数进行特征提取)，操作标签提取
        提取merge文件
        情况1. 当前节点左右孩子都是空，作为edu看待

        #记录EDUs的个数
        N = len(doc.edudict)
        for idx in range(1, N+1, 1):
            node = SpanNode(prop=None)
            node.text = doc.edudict[idx]
            #span的边界确立,包含哪几个EDU形成的跨度
            node.eduspan, node.nucspan = (idx, idx), (idx, idx)
            #当前span关系的核EDU指向
            node.nucedu = idx
            #当前EDUs对应成RST节点进入队列
            self.Queue.append(node)

            队列 和 栈作为全局变量
            
            2017年12月20日发现问题，树和SpanNode数据不一致原因：segmentation得到的edu找回率非100%所以，goldTree的
        构建要通过对标准文件中的标准EDU构建对应merge文件，然后用这些生成文件替代原segmentation得到的文件。
    '''
    def traverse_data_label(self, root):
        # 对遍历的每个状态得到的stack and queue进行特征获取 doc对象传过来
        if root is None:
            return
        else:
            self.traverse_data_label(root.left_child)
            self.traverse_data_label(root.right_child)
            # 对当前两个节点之间的关系，以及当前的组态configuration 进行存储
            fg = FeatureGenerator(self.edu_stack, self.edu_queue, self.doc, self.bcvocab)
            # 取特征
            if self.edu_queue != None:
                print(self.edu_queue[0].text)
            feat = fg.features()
            # 取当前操作,只要当前结点没有孩子节点就对当前节点进行shift, 只要当前节点存在左右孩子就对左右孩子进行reduce,取左右孩子的rel
            # 将孩子出栈，当前节点再入栈
            temp_transition = None
            if root.left_child is None and root.right_child is None:
                # 根据 gold_tree 构建 spanNode
                temp = self.edu_queue.pop(0)
                self.edu_stack.append(temp)
                temp_transition = "Shift"
                print("当前操作：Shift-EDU", self.count)
                self.count += 1
            else:
                # 根据gold tree 提取下面数据
                node_form = root.get_sn_rel()
                node_rel = root.get_rel()
                # input("reduce"+node_form+"   "+node_rel)
                right_node = self.edu_stack.pop()
                left_node = self.edu_stack.pop()
                temp_node = SpanNode(prop=None)
                temp_node.rnode = right_node
                temp_node.lnode = left_node
                # 加前向指针
                temp_node.lnode.pnode, temp_node.rnode.pnode = temp_node, temp_node
                # 节点的文本内容等于合并
                temp_node.text = left_node.text + right_node.text
                print("left text: ", left_node.text)
                print("right text: ", right_node.text)
                if not node_form is None :
                    print("child-NS-from: ", node_form)
                if not node_rel is None :
                    print("child-rel: ", node_rel)
                # span边界
                temp_node.eduspan = (left_node.eduspan[0], right_node.eduspan[1])
                # Nuc span / Nuc EDU 当前Span下的EDUs的组织形式
                temp_node.form = node_form
                if node_form == 'NN':
                    temp_node.nucspan = (left_node.eduspan[0], right_node.eduspan[1])
                    temp_node.nucedu = left_node.nucedu
                    temp_node.lnode.prop = "Nucleus"
                    temp_node.lnode.relation = node_rel
                    temp_node.rnode.prop = "Nucleus"
                    temp_node.rnode.relation = node_rel
                elif node_form == 'NS':
                    temp_node.nucspan = left_node.eduspan
                    temp_node.nucedu = left_node.nucedu
                    temp_node.lnode.prop = "Nucleus"
                    temp_node.lnode.relation = "span"
                    temp_node.rnode.prop = "Satellite"
                    temp_node.rnode.relation = node_rel
                elif node_form == 'SN':
                    temp_node.nucspan = right_node.eduspan
                    temp_node.nucedu = right_node.nucedu
                    temp_node.lnode.prop = "Satellite"
                    temp_node.lnode.relation = node_rel
                    temp_node.rnode.prop = "Nucleus"
                    temp_node.rnode.relation = "span"
                else:
                    raise ValueError("Unrecognized form: {}".format(node_form))
                # 把前两个节点删除之后，把新的节点加入到stack
                self.edu_stack.append(temp_node)
                temp_transition = "Reduce-" +node_form+"-"+node_rel
                print("当前操作 : ", temp_transition)
            print("-------------------------------------------------------------------------")
            # 对当前数据和操作标签进行存储
            self.train_label.append(temp_transition)
            # self.sample_list.append(feat)
    '''
        动态初始化，根据遍历到的文档初始化队列，栈设置为空
    '''

    def init_queue_stack(self):
        self.edu_stack = []
        self.edu_queue = []
        # if not isinstance(self.doc, Doc):
        #     raise ValueError("error_arlen : 对象不是Doc类型")
        # 记录EDUs的个数
        N = len(self.doc.edudict)
        for idx in range(1, N + 1, 1):
            # node = SpanNode(prop=None)
            # node.text = self.doc.edudict[idx]
            # # span的边界确立,包含哪几个EDU形成的跨度
            # node.eduspan, node.nucspan = (idx, idx), (idx, idx)
            # # 当前span关系的核EDU指向
            # node.nucedu = idx
            # # 当前EDUs对应成RST节点进入队列
            # self.edu_queue.append(node)
            self.total_edu_num += 1
        print("初始化Queue中的EDU个数： ",len(self.edu_queue))
        self.total_file_num += 1

    def prep_train_data(self, dis_file_dir, merge_file_dir):
        for file_name in os.listdir(dis_file_dir):
            if file_name.endswith('.dis'):
                self.count = 1
                print("------------处理file_name：", file_name)
                temp_path = os.path.join(dis_file_dir, file_name)
                dis_tree_obj = open(temp_path, 'r')
                # tree_text = dis_tree_obj.read()
                # self.rst_tree.rst_doc_obj = dis_tree_obj
                self.rst_tree.lines_list = dis_tree_obj.readlines()
                root = rst_tree("null")
                # 已优化
                self.rst_tree.create_tree(root, 0)  # 从第一行开始读

                # 获取对应merge对象进行传递
                merge_file_name = os.path.basename(file_name).replace(".out.dis", ".merge")
                merge_file = os.path.join(merge_file_dir, merge_file_name)
                self.doc = self.dr.read(merge_file)
                # 初始化queue stack
                self.init_queue_stack()
                # 带着树 带着doc对象开始后续遍历
                self.traverse_data_label(root)
                # root.pre_traverse(root)
        # 遍历完所有文件和组态与操作之后，存储数据
        self.save_train_data()

    def get_statistic(self):
        print("判断多核关系和二元核心标注")
        print(self.rst_tree.multi_nucleus_rel_statistic)
        print(len(self.rst_tree.multi_nucleus_rel_statistic))
        print(self.rst_tree.nucleus_rel_statistic)
        print("文件总数：", self.total_file_num)
        print("EDU总数：", self.total_edu_num)
        print("每个篇章树平均EDU: ", self.total_edu_num/self.total_file_num)

        print("关系总数：", len(self.rst_tree.rel_static))

    def get_train_vec(self):
        # model里面word embedding 部分还是采用之前的brown cluster
        # pm = ParsingModel(withdp=False, fdpvocab=None, fprojmat=None)
        # for feat in self.sample_list:
        #     configuration_vector = pm.get_vec(feat)
        #     self.train_data.append(configuration_vector)
        pass
    #
    def save_train_data(self):

        file_obj2 = open('/home/arlenzhang/Desktop/Discourse_parser/data/train/train_data_pkl/train_label.pkl', 'wb')
        pkl.dump(self.train_label, file_obj2)
        file_obj3 = open('/home/arlenzhang/Desktop/Discourse_parser/data/train/train_data_pkl/configuration_feature.pkl', 'wb')
        pkl.dump(self.sample_list, file_obj3)
        # 训练数据-获取
        file_obj2.close()
        file_obj3.close()
        print("创建vocab...")
        # 创建vocab
        # vocab_create()
        file_obj1 = open('/home/arlenzhang/Desktop/Discourse_parser/data/train/train_data_pkl/train_data.pkl', 'wb')
        self.get_train_vec()
        pkl.dump(self.train_data, file_obj1)
        file_obj1.close()