import time
import os
import logging
import fcntl

def get_logger(path, suffix):
    cur_time = time.strftime('%Y-%m-%d-%H.%M.%S',time.localtime(time.time()))
    logger = logging.getLogger(__name__+cur_time)
    logger.setLevel(level = logging.INFO)
    handler = logging.FileHandler(os.path.join(path, f"{suffix}_{cur_time}.log"))
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)

    logger.addHandler(handler)
    logger.addHandler(console)
    return logger

class ResultRecorder(object):
    def __init__(self, path, total_cv=10):
        self.path = path # ./logs/utt_fusion_AVL_run2/result.tsv
        self.total_cv = total_cv
        if not os.path.exists(self.path):
            f = open(self.path, 'w')
            f.write('acc\tuar\tf1\n')
            f.close()
    
    def is_full(self, content):
        if len(content) < self.total_cv+1:
            return False
        
        for line in content:
            if not len(line.split('\t')) == 3:
                return False
        return True
    
    def calc_mean(self, content):
        acc = [float(line.split('\t')[0]) for line in content[1:]]
        uar = [float(line.split('\t')[1]) for line in content[1:]]
        f1 = [float(line.split('\t')[2]) for line in content[1:]]
        mean_acc = sum(acc) / len(acc)
        mean_uar = sum(uar) / len(uar)
        mean_f1 = sum(f1) / len(f1)
        return mean_acc, mean_uar, mean_f1

    def write_result_to_tsv(self, results, cvNo):
        # 使用fcntl对文件加锁,避免多个不同进程同时操作同一个文件
        f_in = open(self.path)
        fcntl.flock(f_in.fileno(), fcntl.LOCK_EX) # 加锁
        content = f_in.readlines()
        if len(content) < self.total_cv+1:
            content += ['\n'] * (self.total_cv-len(content)+1)
        content[cvNo] = '{:.4f}\t{:.4f}\t{:.4f}\n'.format(results['acc'], results['uar'], results['f1'])
        if self.is_full(content):
            mean_acc, mean_uar, mean_f1 = self.calc_mean(content)
            content.append('{:.4f}\t{:.4f}\t{:.4f}\n'.format(mean_acc, mean_uar, mean_f1))

        f_out = open(self.path, 'w')
        f_out.writelines(content)
        f_out.close()
        f_in.close()                              # 释放锁
        

