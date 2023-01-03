import os

def make_grid(params):
    total_length = 1
    for key, value in params.items():
        total_length *= len(value)
    
    ans = []
    for _ in range(total_length):
        ans.append({})
    
    combo_num = total_length
    for key, value in params.items():
        combo_num = combo_num // len(value)
        for i in range(0, total_length, combo_num):
            for j in range(combo_num):
                ans[i+j][key] = value[i//combo_num%len(value)]

    return ans

def make_task(parameters):
    # tuned hyper-parameters
    param_grid = make_grid(parameters)
    template = 'sh ' + task_script + ' ' + ' '.join(['{' + key + '}' for key in parameters.keys()])

    total_cmd = []
    for param in param_grid:
        cmd = template.format(**param)
        total_cmd.append(cmd)
    
    # 平均分配gpu
    cmd_with_gpu = []
    for i in range(len(avialable_gpus)):
        task_num = len(total_cmd) / len(avialable_gpus)
        cmds = total_cmd[int(i*task_num):int((i+1)*task_num)]
        for cmd in cmds:
            cmd_with_gpu.append(cmd + ' ' + str(avialable_gpus[i]))
    
    for i in range(num_sessions):
        session_name = '{}_{}'.format(screen_name, i)
        task_file = os.path.join(auto_script_dir, f'{i}_task.sh')
        f = open(task_file, 'w')
        f.write('screen -dmS {}\n'.format(session_name))
        task_num = len(cmd_with_gpu) / num_sessions
        cmds = cmd_with_gpu[int(i*task_num):int((i+1)*task_num)]
        for cmd in cmds:
            _cmd = "screen -x -S {} -p 0 -X stuff '{}\n'\n".format(session_name, cmd)
            f.write(_cmd)
        f.write("screen -x -S {} -p 0 -X stuff 'exit\n'\n".format(session_name))
   
if __name__ == '__main__':
    auto_script_dir = 'auto/tmp'                                # 生成脚本路径
    script_root = 'auto/scripts'
    task_script = script_root + '/' + 'mmin.sh'           # 执行script路径
    avialable_gpus = [0,1,2,3,4,5]                                  # 可用GPU有哪些
    num_sessions = 6                                            # 一共开多少个session同时执行
    avialable_gpus = avialable_gpus[:num_sessions]
    screen_name = 'mmin'
    parameters = {                                              # 一共有哪些参数
        'mse_weight': [0.1, 0.15, 0.2],
        'cycle_weight': [0.05, 0.1, 0.2],
        'run_idx': [1, 2]
    }
    make_task(parameters)

    for i in range(num_sessions):
        cmd = 'sh {}/{}_task.sh'.format(auto_script_dir, i)
        print(cmd)
        os.system(cmd)