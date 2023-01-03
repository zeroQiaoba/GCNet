import os
import numpy as np

def read_results(file):
    ans = []
    lines = open(file).readlines()
    for line in lines:
        if not line.startswith('0'): continue
        ans.append(list(map(lambda x: float(x), line.strip().split('\t'))))
           
    data = np.array(ans).astype(np.float)
    assert data.shape[0] == 24
    val_data = data[0: 10]
    tst_data = data[12: 22]
    return val_data, tst_data

def combine(result1, result2):
    result = result1 * (result1>=result2) + result2 * (result1<result2)
    return result

def combine_file(file1, file2, output):
    val_data1, tst_data1 = read_results(file1)
    val_data2, tst_data2 = read_results(file2)
    val_data = combine(val_data1, val_data2)
    val_mean = np.expand_dims(np.mean(val_data, axis=0), 0)
    val_std = np.expand_dims(np.std(val_data, axis=0), 0)
    val_data = np.vstack([val_data, val_mean, val_std])
    tst_data = combine(tst_data1, tst_data2)
    tst_mean = np.expand_dims(np.mean(tst_data, axis=0), 0)
    tst_std = np.expand_dims(np.std(tst_data, axis=0), 0)
    tst_data = np.vstack([tst_data, tst_mean, tst_std])
    f = open(output, 'w')
    f.write(output.split('/')[-1] + '\n')
    f.write('val:\n')
    for d in val_data:
        line = '\t'.join(list(map(lambda x:'{:.4f}'.format(x), d))) + '\n'
        f.write(line)

    val_mean = val_mean[0]
    val_std = val_std[0]
    acc = '{:.4f}±{:.4f}'.format(val_mean[0], val_std[0])
    uar = '{:.4f}±{:.4f}'.format(val_mean[1], val_std[1])
    f1 = '{:.4f}±{:.4f}'.format(val_mean[2], val_std[2])
    f.write('VAL result:\nacc %s uar %s f1 %s\n\n' % (acc, uar, f1))

    
    f.write('tst:\n')
    for d in tst_data:
        line = '\t'.join(list(map(lambda x:'{:.4f}'.format(x), d))) + '\n'
        f.write(line)
    
    tst_mean = tst_mean[0]
    tst_std = tst_std[0]
    acc = '{:.4f}±{:.4f}'.format(tst_mean[0], tst_std[0])
    uar = '{:.4f}±{:.4f}'.format(tst_mean[1], tst_std[1])
    f1 = '{:.4f}±{:.4f}'.format(tst_mean[2], tst_std[2])
    f.write('TEST result:\nacc %s uar %s f1 %s\n' % (acc, uar, f1))
    
# print(val_data)
# print(tst_data)
root = 'today_tasks/results'
save_root = 'today_tasks/results_combine'
# run_idx1 = 'A_ts_Adnn512,256,128_lossBCE_kd1.0_temp2.0_ce0.0_mmd0.0_run1'
# run_idx2 = 'A_ts_Adnn512,256,128_lossBCE_kd1.0_temp2.0_ce0.0_mmd0.0_run2'
# out = 'A_ts_Adnn512,256,128_lossBCE_kd1.0_temp2.0_ce0.0_mmd0.0'
# combine_file(os.path.join(root, run_idx1), os.path.join(root, run_idx2), os.path.join(save_root, out))
total_file = os.listdir(root)
name_set = set()
for file in total_file:
    name = '_'.join(file.split('_')[:-1])
    name_set.add(name)

for name in name_set:
    run_idx1 = name + '_run1'
    run_idx2 = name + '_run2'
    out = name
    combine_file(os.path.join(root, run_idx1), os.path.join(root, run_idx2), os.path.join(save_root, out))