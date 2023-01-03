import util.classfiy as classfiy
import tensorflow as tf
import numpy as np
from numpy.random import shuffle
from util.util import xavier_init


class CPMNets():
    """build model
    """
    def __init__(self, view_num, trainLen, testLen, layer_size, lsd_dim=128, learning_rate=[0.001, 0.001], lamb=1):
        """
        :param learning_rate:learning rate of network and h
        :param view_num:view number
        :param layer_size:node of each net
        :param lsd_dim:latent space dimensionality
        :param trainLen:training dataset samples
        :param testLen:testing dataset samples
        """
        # initialize parameter
        self.view_num = view_num      # 2
        self.layer_size = layer_size  # [[4096], [4096]]
        self.lsd_dim = lsd_dim        # 512
        self.trainLen = trainLen      # train sample
        self.testLen = testLen        # test sample
        self.lamb = lamb              # 10

        # initialize latent space data
        self.h_train, self.h_train_update = self.H_init('train') # [samplenum, lsd_dim]
        self.h_test, self.h_test_update = self.H_init('test')    # [samplenum, lsd_dim]
        self.h = tf.concat([self.h_train, self.h_test], axis=0)  # [samplenum, lsd_dim]
        self.h_index = tf.placeholder(tf.int32, shape=[None, 1], name='h_index')
        self.h_temp = tf.gather_nd(self.h, self.h_index)  # select sample according to self.h_index

        # initialize the input data: input + sn(mask)
        self.input = dict()
        self.sn = dict()
        for v_num in range(self.view_num):
            self.input[str(v_num)] = tf.placeholder(tf.float32, shape=[None, self.layer_size[v_num][-1]],
                                                    name='input' + str(v_num))
            self.sn[str(v_num)] = tf.placeholder(tf.float32, shape=[None, 1], name='sn' + str(v_num))

        # ground truth
        self.gt = tf.placeholder(tf.int32, shape=[None], name='gt')

        # bulid the model
        self.train_op, self.loss = self.bulid_model([self.h_train_update, self.h_test_update], learning_rate)

        # open session
        gpu_options = tf.GPUOptions(allow_growth=True)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        self.sess.run(tf.global_variables_initializer())

    ## h_update: [self.h_train_update, self.h_test_update]
    ## learning_rate: [0.001, 0.01] (0.001 for recon; 0.01 for all loss)
    def bulid_model(self, h_update, learning_rate):

        ## 在损失计算的时候，输入都是self.h_temp
        ## net: map self.h_temp into view v_num
        net = dict()
        for v_num in range(self.view_num):
            net[str(v_num)] = self.Encoding_net(self.h_temp, v_num)
        reco_loss = self.reconstruction_loss(net)
        imputation_loss = self.imputation_loss(net)
        class_loss = self.classification_loss()
        all_loss = tf.add(reco_loss, self.lamb * class_loss)

        # train net operator [固定h，优化重建过程中的参数]
        # train the network to minimize reconstruction loss
        train_net_op = tf.train.AdamOptimizer(learning_rate[0]) \
            .minimize(reco_loss, var_list=tf.get_collection('weight'))

        ## focus on training data 【固定参数，优化训练过程中的h】
        # train the latent space data to minimize reconstruction loss and classification loss
        train_hn_op = tf.train.AdamOptimizer(learning_rate[1]) \
            .minimize(all_loss, var_list=h_update[0])

        ## focus on testing data 【固定参数，优化训练过程中的h】
        # adjust the latent space data
        adj_hn_op = tf.train.AdamOptimizer(learning_rate[0]) \
            .minimize(reco_loss, var_list=h_update[1])
        return [train_net_op, train_hn_op, adj_hn_op], [reco_loss, class_loss, all_loss, imputation_loss]

    ## initial hidden features
    def H_init(self, a):
        with tf.variable_scope('H' + a):
            if a == 'train':
                h = tf.Variable(xavier_init(self.trainLen, self.lsd_dim))
            elif a == 'test':
                h = tf.Variable(xavier_init(self.testLen, self.lsd_dim))
            h_update = tf.trainable_variables(scope='H' + a)
        return h, h_update

    ## Target: map h into v
    ## h: select from self.h via index, [samplenum, lsd_dim]
    ## v: index of view, int => "self.layer_size[v] is the fc layers of view v"
    def Encoding_net(self, h, v):
        weight = self.initialize_weight(self.layer_size[v])
        layer = tf.matmul(h, weight['w0']) + weight['b0']
        for num in range(1, len(self.layer_size[v])):
            layer = tf.nn.dropout(tf.matmul(layer, weight['w' + str(num)]) + weight['b' + str(num)], 0.9)
        return layer

    ## gain input->latent mapping functions
    def initialize_weight(self, dims_net):
        all_weight = dict()
        with tf.variable_scope('weight'):
            all_weight['w0'] = tf.Variable(xavier_init(self.lsd_dim, dims_net[0])) # [512, 4096]
            all_weight['b0'] = tf.Variable(tf.zeros([dims_net[0]])) # [4096, ]
            tf.add_to_collection("weight", all_weight['w' + str(0)])
            tf.add_to_collection("weight", all_weight['b' + str(0)])
            for num in range(1, len(dims_net)): # if dims_net has two layers, such as [4096, 4096]
                all_weight['w' + str(num)] = tf.Variable(xavier_init(dims_net[num - 1], dims_net[num]))
                all_weight['b' + str(num)] = tf.Variable(tf.zeros([dims_net[num]]))
                tf.add_to_collection("weight", all_weight['w' + str(num)])
                tf.add_to_collection("weight", all_weight['b' + str(num)])
        return all_weight

    ## net: map self.h_temp into view v_num
    ## self.input[str(num)]: origin input for view v_num
    ## self.sn[str(num)]: mask for loss calculation 只重构看到的部分，缺失的部分不管
    def reconstruction_loss(self, net):
        loss = 0
        for v_num in range(self.view_num):
            loss = loss + tf.reduce_sum(
                tf.pow(tf.subtract(net[str(v_num)], self.input[str(v_num)]), 2.0) * self.sn[str(v_num)] # 1(exist) 0(miss)
            )
        return loss

    def imputation_loss(self, net):
        loss = 0
        for v_num in range(self.view_num): # 1(exist) 0(miss)
            view_loss = tf.pow(tf.subtract(net[str(v_num)], self.input[str(v_num)]), 2.0) * (1 - self.sn[str(v_num)])
            loss = loss + tf.reduce_sum(view_loss) / self.layer_size[v_num][-1]
        return loss

    ## 这个和原始论文中计算的方法一样
    ## 注意：分类过程中是没有参数的
    def classification_loss(self):
        ## 计算h_temp中两两维度为512dim，样本之间的相关性
        F_h_h = tf.matmul(self.h_temp, tf.transpose(self.h_temp)) # [samplenum, samplenum]
        F_hn_hn = tf.diag_part(F_h_h) # [samplenum] 返回对角线的值
        F_h_h = tf.subtract(F_h_h, tf.matrix_diag(F_hn_hn)) # [samplenum, samplenum]

        classes = tf.reduce_max(self.gt) - tf.reduce_min(self.gt) + 1
        label_onehot = tf.one_hot(self.gt - 1, classes)  # [samplenum, classes]
        label_num = tf.reduce_sum(label_onehot, 0, keep_dims=True)  # label_num for each class

        F_h_h_sum = tf.matmul(F_h_h, label_onehot)
        label_num_broadcast = tf.tile(label_num, [self.trainLen, 1]) - label_onehot
        F_h_h_mean = tf.divide(F_h_h_sum, label_num_broadcast)

        gt_ = tf.cast(tf.argmax(F_h_h_mean, axis=1), tf.int32) + 1  # gt begin from 1
        F_h_h_mean_max = tf.reduce_max(F_h_h_mean, axis=1, keep_dims=False)

        theta = tf.cast(tf.not_equal(self.gt, gt_), tf.float32)
        F_h_hn_mean_ = tf.multiply(F_h_h_mean, label_onehot) # 按元素相乘
        F_h_hn_mean = tf.reduce_sum(F_h_hn_mean_, axis=1, name='F_h_hn_mean') # F_h_hn_mean 真值对应的距离

        return tf.reduce_sum(tf.nn.relu(tf.add(theta, tf.subtract(F_h_h_mean_max, F_h_hn_mean))))

    # data['0'], data['1']
    # sn: [samplenum, 2]
    # gt: [samplenum, ]
    def train(self, data, sn, gt, epoch, step=[5, 5]):
        global Reconstruction_LOSS
        index = np.array([x for x in range(self.trainLen)])
        shuffle(index)
        sn = sn[index]
        gt = gt[index]
        feed_dict = {self.input[str(v_num)]: data[str(v_num)][index] for v_num in range(self.view_num)}
        feed_dict.update({self.sn[str(i)]: sn[:, i].reshape(self.trainLen, 1) for i in range(self.view_num)})
        feed_dict.update({self.gt: gt})
        feed_dict.update({self.h_index: index.reshape((self.trainLen, 1))})
        for iter in range(epoch):
            # update the network
            for i in range(step[0]):
                _, Reconstruction_LOSS, Classification_LOSS = self.sess.run(
                    [self.train_op[0], self.loss[0], self.loss[1]], feed_dict=feed_dict)
            # update the h
            for i in range(step[1]):
                _, Reconstruction_LOSS, Classification_LOSS = self.sess.run(
                    [self.train_op[1], self.loss[0], self.loss[1]], feed_dict=feed_dict)
            output = "Epoch : {:.0f}  ===> Reconstruction Loss = {:.4f}, Classification Loss = {:.4f} " \
                .format((iter + 1), Reconstruction_LOSS, Classification_LOSS)
            print(output)

    ## input['0']; input['1']; sn['0']; sn['1']
    ## h_index: [a, a+1, ...., a+n] => 默认train在test前面
    def test(self, data, sn, gt, epoch):
        feed_dict = {self.input[str(v_num)]: data[str(v_num)] for v_num in range(self.view_num)}
        feed_dict.update({self.sn[str(i)]: sn[:, i].reshape(self.testLen, 1) for i in range(self.view_num)})
        feed_dict.update({self.gt: gt})
        feed_dict.update({self.h_index: np.array([x for x in range(self.testLen)]).reshape(self.testLen, 1) + self.trainLen})

        # update the h
        for iter in range(epoch):
            for i in range(5):
                _, Reconstruction_LOSS, Imputation_LOSS = self.sess.run(
                    [self.train_op[2], self.loss[0], self.loss[3]], feed_dict=feed_dict)
            output = "Epoch : {:.0f}  ===> Reconstruction Loss = {:.4f}" \
                .format((iter + 1), Reconstruction_LOSS)
            print(output)
        return Imputation_LOSS


    def get_h_train(self):
        lsd = self.sess.run(self.h)
        return lsd[0:self.trainLen]


    def get_h_test(self):
        lsd = self.sess.run(self.h)
        return lsd[self.trainLen:]

