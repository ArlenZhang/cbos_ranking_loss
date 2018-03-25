""" Call a pretrained segmenter for discourse segmentation
"""
import code.discoseg.buildedu as buildedu
from os.path import join, basename
# from code.discoseg.edu_repre_model.docreader import DocReader
import os
# from sys import argv
import re

# 默认对文件同目录修改
def seg(fmodel_p, fvocab_p, read_p):
    buildedu.main(fmodel_p, fvocab_p, read_p, read_p)

# 默认对文件同目录修改
def push_edus_file(read_p):
    write_p = read_p
    flist = [join(read_p, fname) for fname in os.listdir(read_p) if fname.endswith('merge')]
    for (fidx, fname) in enumerate(flist):
        print('fname: ', fname)
        last_index = 1
        last_str = ""
        f_obj = open(fname, 'r')
        f_name = basename(fname).replace(".merge", ".edus")
        f_name = join(write_p, f_name)
        write_lines = list()
        # 获取
        for line in f_obj:
            # result = re.search(r'[\w\W]*([\d]+)$',line)
            if len(line) <= 1:
                continue
            result = re.search(r'[\d]+[\s]+[\d]+[\s]+([\w,.-]+)[\s]+[\w\W]+[\s]+([\d]+)$', line)

            if not result is None:
                print(result.group(1))
                temp = int(result.group(2))
                print(temp, '============', last_index)
                if temp == last_index:
                    last_str += result.group(1) + " "
                else:
                    write_lines.append(last_str + "\n")
                    last_str = result.group(1) + " "
                    last_index += 1
        write_lines.append(last_str)
        print(write_lines)
        # 存储 last_str
        with open(f_name, 'a') as f:
            f.writelines(write_lines)
