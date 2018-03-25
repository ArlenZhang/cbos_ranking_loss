"""
    作者：张龙印
    日期：2018.3.12
    文本处理
    文本处理
"""
# 定义英文符号
s_list = [',', '.', ':', ';', '!', '?', '-', '*', '\'', '_', '\"', '(', ')', '{', '}', '[', ']', '<', '>',
          '¨', '"', '||', '/', '&', '~', '$', '\\', '#']


# 去除一个词两边的特殊符号, 当然不去这些符号和去两种训练词向量的方案都要尝试，边缘特征
def rm_edge_s(word):
    word = word.strip()
    first, last = word[0], word[-1]
    while first in s_list:
        word = word[1:]
        if len(word) > 0:
            first = word[0]
        else:
            break
    while last in s_list:
        word = word[:-1]
        if len(word) > 0:
            last = word[-1]
        else:
            break
    return word
