from __future__ import print_function
from __future__ import division

import tensorflow as tf
import framework.model.trntst


def TrnTst(framework.model.trntst.TrnTst):

  def _construct_feed_dict_in_trn(self, data):
    raise NotImplementedError("""please customize construct_feed_dict_in_trn""")

  # return loss value  
  def feed_data_and_run_loss_op_in_val(self, data, sess):
    raise NotImplementedError("""please customize feed_data_and_run_loss_op_in_val""")

  # add eval result to metrics dictionary, key is metric name, val is metric value
  def predict_and_eval_in_val(self, sess, tst_reader, metrics):
    raise NotImplementedError("""please customize predict_and_eval_in_val""")

  # write predict result to predict_file
  def predict_in_tst(self, sess, tst_reader, predict_file):
    raise NotImplementedError("""please customize predict_in_tst""")

  def _iterate_epoch(self, sess, trn_reader, tst_reader, 
    summarywriter, step, total_step, epoch):
    
    trn_batch_size = self.model_cfg.trn_batch_size
    avg_trn_loss = 0.
    batches_per_epoch = 0

    for data in trn_reader.yield_trn_batch(trn_batch_size):
      if self.model_cfg.monitor_iter > 0 and step % self.model_cfg.monitor_iter == 0:
        self.feed_data_and_monitor(data, sess, step)

      loss_value = self.feed_data_and_trn(data, sess, summarywriter=summarywriter, step=step)
      # print('step', step, 'loss', loss_value)
      avg_trn_loss += loss_value
      batches_per_epoch += 1
      
      step += 1

      if self.model_cfg.summary_iter > 0 and step % self.model_cfg.summary_iter == 0:
        summarystr = self.feed_data_and_summary(data, sess)
        summarywriter.add_summary(summarystr, step)

      if self.model_cfg.val_iter > 0 and step % self.model_cfg.val_iter == 0:
        metrics = self._validation(sess, tst_reader)
        metrics_str = 'step (%d/%d) '%(step, total_step)
        for key in metrics:
          metrics_str += '%s:%.4f '%(key, metrics[key])
        self._logger.info(metrics_str)

    self.model.saver.save(
      sess, os.path.join(self.path_cfg.model_dir, 'epoch'), global_step=epoch)

    avg_trn_loss /= batches_per_epoch
    return step, avg_trn_loss


