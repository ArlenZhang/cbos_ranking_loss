from os import listdir
from os.path import join, basename
from code.discoseg.model.classifier import Classifier
from code.discoseg.model.docreader import DocReader
from code.discoseg.model.sample import SampleGenerator
# from discoseg.cPickle import load
import gzip
import _pickle as cPickle
import multiprocessing as mp

rpath_ = ""
wpath_ = ""
clf_ = Classifier()
vocab_ = None
dr_ = None
# from sklearn import svm
def main(fmodel, fvocab, rpath, wpath):
    global rpath_, wpath_, clf_, vocab_, dr_
    rpath_ = rpath
    wpath_ = wpath
    clf_.loadmodel(fmodel)
    vocab_ = cPickle.load(gzip.open(fvocab))
    dr_ = DocReader()
    flist = [join(rpath, fname) for fname in listdir(rpath) if fname.endswith('conll')]
    pool = mp.Pool(processes=4)
    pool.map(do_seg_one_, flist)

def do_seg_one_(fname):
    doc = dr_.read(fname, withboundary=False)
    sg = SampleGenerator(vocab_)
    sg.build(doc)
    M, _ = sg.getmat()
    predlabels = clf_.predict(M)
    doc = postprocess(doc, predlabels)
    writedoc(doc, fname, wpath_)

def postprocess(doc, predlabels):
    """ Assign predlabels into doc
    """
    tokendict = doc.tokendict
    for gidx in tokendict.keys():
        if predlabels[gidx] == 1:
            tokendict[gidx].boundary = True
        else:
            tokendict[gidx].boundary = False
        if tokendict[gidx].send:
            tokendict[gidx].boundary = True
    return doc


# def writedoc(doc, fname, wpath):
#     """ Write doc into a file with the CoNLL-like format
#     """
#     tokendict = doc.tokendict
#     N = len(tokendict)
#     fname = basename(fname) + '.edu'
#     fname = join(wpath, fname)
#     eduidx = 0
#     with open(fname, 'w') as fout:
#         for gidx in range(N):
#             fout.write(str(eduidx) + '\n')
#             if tokendict[gidx].boundary:
#                 eduidx += 1
#             if tokendict[gidx].send:
#                 fout.write('\n')
#     print 'Write segmentation: {}'.format(fname)


def writedoc(doc, fname, wpath):
    """ Write file
    """
    tokendict = doc.tokendict
    N = len(tokendict)
    fname = basename(fname).replace(".conll", ".merge")
    fname = join(wpath, fname)
    eduidx = 1
    with open(fname, 'w') as fout:
        for gidx in range(N):
            tok = tokendict[gidx]
            line = str(tok.sidx) + "\t" + str(tok.tidx) + "\t"
            line += tok.word + "\t" + tok.lemma + "\t"
            line += tok.pos + "\t" + tok.deplabel + "\t"
            line += str(tok.hidx) + "\t" + tok.ner + "\t"
            # 为了程序能继续走下去 修改了代码 张龙印 log
            if tok.partialparse is None:
                line += 'None' + "\t" + str(eduidx) + "\n"
            else:
                line += tok.partialparse + "\t" + str(eduidx) + "\n"
            # print("----->",type(tok.partialparse),'  eduindex-->',eduidx)
            fout.write(line)
            # Boundary
            if tok.boundary:
                eduidx += 1
            if tok.send:
                fout.write("\n")
