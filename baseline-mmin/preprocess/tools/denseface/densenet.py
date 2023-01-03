from __future__ import print_function
from __future__ import division

import collections
import tensorflow as tf
slim = tf.contrib.slim

import framework.model.proto

class DenseNetConfig(framework.model.proto.ProtoConfig):
  def __init__(self):
    self.bc_mode = True  # DenseNet, DenseNet-BC
    self.growth_rate = 12 # Growth rate for every dense layer
    self.depth = 40 # the depth of whole network
    self.total_blocks = 3
    self.keep_prob = 1.0
    self.weight_decay = 1e-4
    self.reduction = 0.5

    self.image_size = 64
    self.num_chanels = 1
    self.num_classes = 8

    # automatic
    self.first_output_features = self.growth_rate * 2
    self.layers_per_block = (self.depth - 1 - self.total_blocks) // self.total_blocks
    if self.bc_mode:
      self.layers_per_block = self.layers_per_block // 2


class DenseNetProto(framework.model.proto.ModelProto):
  name_scope = 'DenseNetProto'

  def __init__(self, config):
    super(DenseNetProto, self).__init__(config)
    # inputs
    self.images = tf.no_op()
    self.labels = tf.no_op()
    self.learning_rate = tf.no_op()
    self.is_training = tf.no_op()
    # outputs
    self.end_points = collections.OrderedDict()
    self.accuracy = tf.no_op()

  def build_parameter_graph(self, basegraph):
    pass

  def densenet_arg_scope(is_training):
    with slim.arg_scope([slim.conv2d], padding='SAME', strides=[1, 1, 1, 1], 
      activation_fun=tf.nn.relu,
      weights_initializer=tf.contrib.layers.variance_scaling_initializer()):
      with slim.arg_scope([slim.batch_norm], center=True, scale=True, is_training=is_training):
        with slim.arg_scope([slim.avg_pool2d], kernel_size=2, stride=2, padding='VALID'):
          with slim.arg_scope([slim.dropout], keep_prob=self.config.keep_prob,
            is_training=is_training) as arg_sc:
            return arg_sc

  def composite_function(self, _input, out_channels, kernel_size=3, 
    name_scope='composite_function'):
    """Function from paper H_l that performs:
    - batch normalization
    - ReLU nonlinearity
    - convolution with required kernel
    - dropout, if required
    """
    with tf.variable_scope(name_scope):
      # BN
      output = slim.batch_norm(_input)
      # ReLU
      output = tf.nn.relu(output)
      # convolution
      output = slim.conv2d(output, out_channels, kernel_size)
      # dropout
      output = slim.dropout(output)
    return output


  def add_internal_layer(self, _input, growth_rate):
    """Perform H_l composite function for the layer and after concatenate
    input with output from composite function.
    """
    # call composite function with 3x3 kernel
    if not self.config.bc_mode:
      comp_out = self.composite_function(_input, growth_rate, 3)
    else:
      bottleneck_out = self.composite_function(_input, 4*growth_rate, 1, 
        name_scope='bottleneck')
      comp_out = self.composite_function(bottleneck_out, growth_rate, 3)
    output = tf.concat(values=(_input, comp_out), axis=3)
    return output

  def add_block(self, _input, growth_rate, layers_per_block):
    """Add N H_l internal layers"""
    net = _input
    for layer in xrange(layers_per_block):
      with tf.variable_scope('layer_%d' % layer):
        net = self.add_internal_layer(net, growth_rate)
    return net

  def transition_layer(self, _input):
    """Call H_l composite function with 1x1 kernel and after average
    pooling
    """
    # 1x1 kernel conv
    out_channels = int(int(_input.get_shape()[-1]) * self.reduction)
    output = self.composite_function(_input, out_channels, 1)
    # average pooling
    output = slim.avg_pool2d(output)
    return output

  def build_inference_graph_in_trn_tst(self, basegraph):
    with basegraph.as_default():
      with tf.variable_scope(self.name_scope):
        arg_scope = self.densenet_arg_scope(self.is_training):
        with slim.arg_scope(arg_scope):
          # first: initial 3x3 conv to first_output_features
          # 32x32
          net = slim.conv2d(self.images, 
            self.proto_config.first_output_features,
            3, strides=[1, 2, 2, 1],
            name='Initial_convolution')

          # addd N required blocks
          for block in xrange(self.config.total_blocks):
            with tf.variable_scope('Block_%d' % block):
              net = self.add_block(net, self.config.growth_rate,
                self.config.layers_per_block)
              # last block without transition layer
              if block != self.config.total_blocks - 1:
                with tf.variable_scope('Transition_after_block_%d' % block):
                  net = self.transition_layer(net)

          # last layer
          with tf.variable_scope('Transition_to_classes'):
            net = slim.batch_norm(net)
            net = tf.nn.relu(net)
            last_pool_channel = int(net.get_shape()[-2])
            net = slim.avg_pool2d(net, kernel_size=last_pool_channel,
              stride=last_pool_channel)
            net_flatten = tf.reshape(net, (-1, net.get_shape()[-1]))
            self.end_points['last_pool'] = net_flatten
            logits = slim.fully_connected(net, self.config.num_classes,
              activation_fn=None, 
              weights_initializer=tf.contrib.layers.xavier_initializer(),
              biases_initializer=tf.zeros_initializer())
            self.end_points['logits'] = logits
          
          preds = tf.nn.softmax(logits)
          self.end_points['preds'] = preds

          correct_prediction = tf.equal(
            tf.argmax(preds, 1), tf.argmax(self.labels, 1))
          self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32),
            name='accuracy_per_batch')

          self.append_op2monitor('accuracy_per_batch', self.accuracy)



  def build_inference_graph_in_tst(self, basegraph):
    self.build_inference_graph_in_trn_tst(basegraph)


class DenseNetModel(framework.model.proto.FullModel):
  name_scope = 'DenseNetModel'

  def get_model_proto(self):
    return DenseNet(self.config.proto_config)
  
  def add_tst_input(self, basegraph):
    proto_config = self.config.proto_config
    with basegraph.as_default():
      with tf.variable_scope(self.name_scope):
        self.images = tf.placeholder(tf.float32, 
          shape=(None, proto_config.image_size, proto_config.image_size,
            proto_config.num_chanels), name='images')
        self.is_training = tf.placeholder(tf.bool, shape=[], name='is_training')

  def add_trn_tst_input(self, basegraph):
    proto_config = self.config.proto_config
    with basegraph.as_default():
      with tf.variable_scope(self.name_scope):
        self.images = tf.placeholder(tf.float32, 
          shape=(None, proto_config.image_size, proto_config.image_size,
            proto_config.num_chanels), name='images')
        self.labels = tf.placeholder(tf.float32,
          shape=(None, proto_config.num_classes), name='labels')
        self.learning_rate = tf.placeholder(tf.float32,
          shape=[], name='learning_rate')
        self.is_training = tf.placeholder(tf.bool, shape=[], name='is_training')

  def add_loss(self, basegraph):
    with basegraph.as_default():
      with tf.variable_scope(self.name_scope):
        self.cross_entropy = tf.reduce_mean(
          tf.nn.softmax_cross_entropy_with_logits(
            logits=self.model_proto.end_points['logits'],
            labels=self.labels))
        self.append_op2monitor('train_cross_entropy', self.cross_entropy)
        weight_l2_loss = tf.add_n(
          [tf.nn.l2_loss(var) for var in tf.trainable_variables()])

        weight_decay = self.config.proto_config.weight_decay 
        loss_op = self.cross_entropy + weight_decay * weight_l2_loss
        return loss_op


  def build_trn_tst_graph(self):
    basegraph = tf.Graph()
    self._trn_tst_graph = basegraph

    self._build_parameter_graph(basegraph)
    self.add_trn_tst_input(basegraph)

    self._build_inference_graph_in_trn_tst(basegraph)

    self._loss_op = self.add_loss(basegraph)

    for key, value in self._model_proto.op2monitor.iteritems():
      self._op2monitor[key] = value

    with basegraph.as_default():
      with tf.variable_scope(self.name_scope):
        
        self.learning_rates = [self.learning_rate]

        if self.config.optimizer_alg == 'Adam':
          optimizer = tf.train.AdamOptimizer(self.learning_rate)
        elif self.config.optimizer_alg == 'SGD':
          optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        elif self.config.optimizer_alg == 'Momentum':
          optimizer = tf.train.MomentumOptimizer(
            self.learning_rate, 0.9, use_nesterov=True)

        self._gradient_op = optimizer.compute_gradients(self._loss_op)
        self._train_op = optimizer.apply_gradients(self._gradient_op, global_step=global_step)

    self._train_ops = [self._train_op]
    self._add_saver(basegraph)
    self._add_summary(basegraph)
    self._add_init(basegraph)

    return basegraph

