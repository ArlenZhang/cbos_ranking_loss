# 张龙印
# 对PDTB文件的xml格式生成， 转移到converted_corpus/pdtb

scriptdir=`dirname $0`
scriptdir='/home/arlenzhang/stanford/stanford-corenlp-full-2015-12-09'
echo $scriptdir

# $1 - path
DPLP_PATH='/home/arlenzhang/Desktop/discourse_parse_r2_pro/*.out'
raw='/home/arlenzhang/Desktop/discourse_parse_r2_pro/data/corpus/PDTB/raw'
pdtb='/home/arlenzhang/Desktop/discourse_parse_r2_pro/data/converted_corpus/pdtb'

# 训练数据
for FNAME in $raw/*
do
    /usr/bin/java -mx2g -cp "$scriptdir/*" edu.stanford.nlp.pipeline.StanfordCoreNLP -annotators tokenize,ssplit,pos,lemma,ner,parse -ssplit.eolonly -tokenize.whitespace true -file $FNAME
    /bin/mv $DPLP_PATH $pdtb
done