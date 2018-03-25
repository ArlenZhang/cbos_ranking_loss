"""
建立一个可以操作的单独的rst树类
    给左右孩子 当前节点
    给根据文档wsj的dis文件建立树的方法，返回根root
    ( Root (span 1 3)
      ( Nucleus (leaf 1) (rel2par span) (text _!Spencer J. Volk, president and chief operating officer of this consumer and industrial products company, was elected a director._!) )
      ( Satellite (span 2 3) (rel2par elaboration-additional)
        ( Nucleus (leaf 2) (rel2par span) (text _!Mr. Volk, 55 years old, succeeds Duncan Dwight,_!) )
        ( Satellite (leaf 3) (rel2par elaboration-additional-e) (text _!who retired in September._!) )
      )
    )
    运用前序遍历的方法就可以创建，只要遍历过程中发现是span就作为node递归下去，标记行数
    
    问题： 关系抽取中遇到关系怎么是结构关系，因为之前DPLP训练出来的操作对应的关系是内容深层次的关系
单个对象就是tree node 
    作者： 张龙印
    日期： 2018.3.11
"""
import re

# 定义模式匹配
leaf_parttern = r' *\( \w+ \(leaf .+'
leaf_re = re.compile(leaf_parttern)
node_parttern = r' *\( \w+ \(span .+'
node_re = re.compile(node_parttern)
end_parttern = r'\s*\)\s*'
end_re = re.compile(end_parttern)
nodetype_parttern = r' *\( (\w+) .+'
type_re = re.compile(nodetype_parttern)
rel_parttern = r' *\( \w+ \(.+\) \(rel2par ([\w-]+).+'
rel_re = re.compile(rel_parttern)
node_leaf_parttern = r' *\( \w+ \((\w+) \d+.*\).+'
node_leaf_re = re.compile(node_leaf_parttern)

# 公共参数
edus_list = []

def get_blank(line):
    count = 0
    while line[count] == " ":
        count += 1
    return count


class rst_tree:
    def __init__(self, type_=None, l_ch=None, r_ch=None, p_node=None, child_rel=None, rel=None, ch_ns_rel="",
                 lines_list=None, temp_line=" "):
        # 类型为三种 Satellite Nucleus Root
        self.type = type_
        self.left_child = l_ch
        self.right_child = r_ch
        self.parent_node = p_node
        # 左右孩子节点的关系
        self.child_rel = child_rel
        # 当前结点上标注的rel
        self.rel = rel
        # 左右孩子的卫星关系
        self.child_NS_rel = ch_ns_rel
        # 动态获取的文档对象
        self.lines_list = lines_list
        # temp_line当前节点在文件中的开始描述
        self.temp_line = temp_line
        # 当前span的拼接
        self.temp_edu = None
        self.temp_edu_ids = None

    def append(self, root):
        pass

    def get_type(self, line):
        pass

    '''
        以当前行数为起点的一个节点树的构建，这样保证了递归实现的可行性
        张龙印 2017年12月15日
    '''

    def create_tree(self, temp_line_num, p_node_=None):
        if temp_line_num > len(self.lines_list):
            return
        if self.type == "null":
            line = self.lines_list[temp_line_num]
            self.temp_line = line
            self.type = "Root"
            temp_line_num += 1
        line = self.lines_list[temp_line_num]
        # 为了判断当前line的父节点是否存在第三个孩子，我们在第一个孩子这里取空串长度作为哨
        count_blank = get_blank(line)
        # 创建一个rool_list[] 存储当前所有的孩子节点
        root_list = []
        while temp_line_num < len(self.lines_list):
            line = self.lines_list[temp_line_num]
            if get_blank(line) == count_blank:  # 下面一行还是孩子节点
                temp_line_num += 1
                node_type = type_re.findall(line)[0]
                root_new = rst_tree(type_=node_type, temp_line=line, rel=rel_re.findall(line)[0],
                                    lines_list=self.lines_list)
                # 是节点就继续
                if node_re.match(line):
                    temp_line_num = root_new.create_tree(temp_line_num=temp_line_num, p_node_=root_new)

                elif leaf_re.match(line):
                    pass
                # 结点入栈
                root_list.append(root_new)
            else:
                # 下一行不是孩子节点
                # 对最右端孩子之后进行 行递增
                while temp_line_num < len(self.lines_list) and end_re.match(self.lines_list[temp_line_num]):
                    temp_line_num += 1
                break

        # 对当前root_list[]里面的所有孩子节点和p_root关联起来
        while len(root_list) > 2:
            temp_r = root_list.pop()
            temp_l = root_list.pop()
            if not temp_l.rel == "span":
                new_node = rst_tree(type_="Nucleus", r_ch=temp_r, l_ch=temp_l, ch_ns_rel="NN", child_rel=temp_l.rel,
                                    rel=temp_l.rel, temp_line="<new created line>", lines_list=self.lines_list)
            if not temp_r.rel == "span":
                new_node = rst_tree(type_="Nucleus", r_ch=temp_r, l_ch=temp_l, ch_ns_rel="NN", child_rel=temp_r.rel,
                                    rel=temp_r.rel, temp_line="<new created line>", lines_list=self.lines_list)
            new_node.temp_line = "<new created line>"

            # 指向父节点
            temp_r.parent_node = new_node
            temp_l.parent_node = new_node

            # 创建结束，将新结点入栈
            root_list.append(new_node)

        self.right_child = root_list.pop()
        self.left_child = root_list.pop()
        # 指向父节点
        self.right_child.parent_node = p_node_
        self.left_child.parent_node = p_node_

        # 孩子rel关系
        if not self.right_child.rel == "span":
            self.child_rel = self.right_child.rel
        if not self.left_child.rel == "span":
            self.child_rel = self.left_child.rel

        # 孩子NS关系
        self.child_NS_rel += self.left_child.type[0] + self.right_child.type[0]

        return temp_line_num

    '''
        验证树建立的合法性，进行前序遍历
            1希望建立的树和文件保持一致
            2希望shift-reduce过程和configuration保持同步，shift-reduce过程和文件的
        后续遍历保持一致，验证。
    '''

    def pre_traverse(self):
        print("当前节点的文件描述：", self.temp_line)
        print("当前节点的EDU: ", self.temp_edu)
        print("当前节点的EDU_ids: ", self.temp_edu_ids)
        if self.parent_node is not None:
            print("当前节点的父节点：", self.parent_node.temp_line)
        else:
            print("父节点为空")
        print("temp_type ", self.type, 'child_rel ', self.child_rel)
        print("----------")
        if self.right_child is not None and self.left_child is not None:
            self.left_child.pre_traverse()
            self.right_child.pre_traverse()

    # 将当前文件的edu加入edus_list
    def pre_traverse_leaves(self, temp_node, temp_edu_file, temp_eduids_file):
        if temp_node.type == "Root":
            edus_list.clear()

        if self.right_child is not None and self.left_child is not None:
            self.left_child.pre_traverse_leaves(self.left_child, temp_edu_file, temp_eduids_file)
            self.right_child.pre_traverse_leaves(self.right_child,  temp_edu_file, temp_eduids_file)
            # 当前节点不是叶子节点，edus取孩子节点的拼接结果
            self.temp_edu = self.left_child.temp_edu + self.right_child.temp_edu
            self.temp_edu_ids = self.left_child.temp_edu_ids + self.right_child.temp_edu_ids

        elif self.right_child is None and self.left_child is None:
            # 当前节点是叶子节点，edus取当前节点的edu本身即可
            self.temp_edu = temp_edu_file.pop(0)
            self.temp_edu_ids = temp_eduids_file.pop(0)
            edus_list.append(temp_node)
