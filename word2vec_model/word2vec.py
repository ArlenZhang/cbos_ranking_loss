"""
    word2vec skip-gram model with NCE loss and
    这套代码似乎更合理，运用create xxx 将一些函数过程传递给类，作为属性
    问题：怎么运行两套学习率代码比较曲线 toggle run
    run tensorboard --logdir='/home/arlenzhang/Desktop/Workstation/Period2/visualization'
        tensorboard --logdir=name1:'/home/arlenzhang/Desktop/Workstation/Period2/graphs/word2vec/lr1.0',
        name2:'/home/arlenzhang/Desktop/Workstation/Period2/graphs/word2vec/lr0.5'

    Word2Vec模型中，主要有Skip-Gram和CBOW两种模型，从直观上理解，Skip-Gram是给定input word来预测上下文。
    而CBOW是给定上下文，来预测input word。本篇文章仅讲解Skip-Gram模型。
"""
from code.utils.word2vec_utils import *
from code.utils.file_processor import *
from code.word2vec_model.model_config import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'

class SkipGramModel:
    """ Build the graph for word2vec model """

    def __init__(self, dataset, vocab_size, embed_size, batch_size, num_sampled, learning_rate):
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.batch_size = batch_size
        self.num_sampled = num_sampled
        self.lr = learning_rate
        self.global_step = tf.get_variable('global_step_w', initializer=tf.constant(0), trainable=False)
        self.skip_step = SKIP_STEP
        self.dataset = dataset
        # 定义权重和偏移量
        self.nce_weight = tf.get_variable('nce_weight',
                                          shape=[self.vocab_size, self.embed_size],
                                          initializer=tf.truncated_normal_initializer(
                                              stddev=1.0 / (self.embed_size ** 0.5))
                                          )
        self.nce_bias = tf.get_variable('nce_bias', initializer=tf.zeros([VOCAB_SIZE]))
        # 会话和绘图路径
        if CORPUS_TYPE is "Gigaword":
            self.lr_p = 'graphs/w2v_giga/lr'
        else:
            self.lr_p = 'graphs/w2v_pdtb/lr'

    def _import_data(self):
        """
            Step 1: import data
        """
        with tf.name_scope('data'):
            self.iterator = self.dataset.make_initializable_iterator()
            self.center_words, self.target_words = self.iterator.get_next()

    def _create_embedding(self):
        """
            Step 2 : embedding lookup.
            In word2vec, it's actually the weights that we care about
        """
        with tf.name_scope('embed'):
            self.embed_matrix = tf.get_variable('embed_matrix',
                                                shape=[self.vocab_size, self.embed_size],
                                                initializer=tf.random_uniform_initializer())
            self.embed = tf.nn.embedding_lookup(self.embed_matrix, self.center_words, name='embedding')

    def _create_loss(self):
        with tf.name_scope('loss'):
            """ Step 3: define the loss function """
            self.loss = tf.reduce_mean(tf.nn.nce_loss(weights=self.nce_weight,
                                                      biases=self.nce_bias,
                                                      labels=self.target_words,
                                                      inputs=self.embed,
                                                      num_sampled=self.num_sampled,
                                                      num_classes=self.vocab_size), name='loss')

    def _create_optimizer(self):
        """ Step 5: define optimizer """
        self.optimizer = tf.train.GradientDescentOptimizer(self.lr).minimize(self.loss,
                                                                             global_step=self.global_step)

    def _create_summaries(self):
        with tf.name_scope('summaries'):
            tf.summary.scalar('loss', self.loss)
            tf.summary.histogram('histogram loss', self.loss)
            # because you have several summaries, we should merge them all
            # into one op to make it easier to manage
            self.summary_op = tf.summary.merge_all()

    def build_graph(self):
        """
            Build the graph for our model
            建立模型大概就这几个模块分布考虑一下
        """
        self._import_data()
        self._create_embedding()
        self._create_loss()
        self._create_optimizer()
        self._create_summaries()

    def train(self, num_train_steps):
        with tf.Session() as sess:
            # dataset 这套代码需要初始化iterator迭代器
            sess.run(self.iterator.initializer)
            # 全局初始化
            sess.run(tf.global_variables_initializer())
            # 学习和训练的过程
            total_loss = 0.0  # we use this to calculate late average loss in the last SKIP_STEP steps
            writer = tf.summary.FileWriter(self.lr_p + str(self.lr), sess.graph)
            initial_step = self.global_step.eval()
            for index in range(initial_step, initial_step + num_train_steps):
                try:
                    loss_batch, _, summary = sess.run([self.loss, self.optimizer, self.summary_op])
                    writer.add_summary(summary, global_step=index)
                    total_loss += loss_batch
                    if (index + 1) % self.skip_step == 0:
                        print('Average loss at step {}: {:5.1f}'.format(index, total_loss / self.skip_step))
                        total_loss = 0.0
                except tf.errors.OutOfRangeError:
                    # 训练次数很多，每次迭代128个新的数据进来，当读完所有训练数据训练还没玩成就会初始化迭代器重头
                    sess.run(self.iterator.initializer)
            final_embed_matrix = sess.run(self.embed_matrix)
            update_embedding_matrix(final_embed_matrix)
            writer.close()

# 注意： 第一层得到的是dict[word] = vector
def update_embedding_matrix(final_m):
    if CORPUS_TYPE == "Gigaword":
        dictionary = load_data(Giga_WORD2IDS_PKL)
        result_embedding = dict()
        for word in dictionary.keys():
            result_embedding[word] = final_m[dictionary[word]]
        # 写入
        save_data(result_embedding, Giga_EMBEDDING_PKL)
    else:
        dictionary = load_data(PDTB_WORD2IDS_PKL)
        result_embedding = dict()
        for word in dictionary.keys():
            result_embedding[word] = final_m[dictionary[word]]
        # 写入
        save_data(result_embedding, PDTB_EMBEDDING_PKL)

def gen():
    yield from batch_gen(BATCH_SIZE, SKIP_WINDOW, use_edus=True)


def main_word():
    dataset = tf.data.Dataset.from_generator(gen, (tf.int32, tf.int32),
                                             (tf.TensorShape([BATCH_SIZE]), tf.TensorShape([BATCH_SIZE, 1])))
    # giga or model
    model = SkipGramModel(dataset, VOCAB_SIZE, EMBED_SIZE, BATCH_SIZE, NUM_SAMPLED, LEARNING_RATE)
    model.build_graph()
    model.train(NUM_TRAIN_STEPS)
