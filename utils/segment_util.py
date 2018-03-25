"""
    切割部分函数共享

"""
from code.utils.xmlreader import reader, writer, combine
import os
import multiprocessing as mp

# 对xml文件进行数据提取
def extract(rpath):
    files = [os.path.join(rpath, fname) for fname in os.listdir(rpath) if fname.endswith(".out")]
    pool = mp.Pool(processes=4)
    pool.map(one_extract, files)

def one_extract(file_one):
    sentlist, constlist = reader(file_one)
    sentlist = combine(sentlist, constlist)
    fconll = file_one.replace(".out", ".conll")
    writer(sentlist, fconll)
