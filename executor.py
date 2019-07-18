# This file is to execute deep learning jobs
import os
import sys
from scheduler import Scheduler
from queue import Queue
import threading


class Executor:
    def __init__(self, daemon_server=None):
        self.execute_queue = Queue()

    def execute(self, job_info):
        if job_info['model'] == 'resnet':
            self.execute_resnet(job_info)
        elif job_info['model'] == 'inception':
            self.execute_inception(job_info)
        elif job_info['model'] == 'vgg':
            self.execute_vgg(job_info)

    def execute_resnet(self, job_info):
        print(job_info)
        node = job_info['gpu_info'][0]
        gpu = job_info['gpu_info'][1]
        ssh = 'ssh ' + node + ' '
        gpu_pre = 'env CUDA_VISIBLE_DEVICES=' + str(gpu)
        conda = 'conda activate stable; '
        source = 'source .zshrc; source ~/configure/scheduler1; '
        cd = 'cd scheduler/models-1.11/official/resnet; '
        exec_cmd = gpu_pre + ' python cifar10_main.py --md=' + job_info['loc'] +\
                   ' -lr=' + str(job_info['hyparams'][0]) +\
                   ' -bs=' + str(int(job_info['hyparams'][1])) +\
                   ' -te=' + str(job_info['hyparams'][2]) +\
                   ' -status=' + str(job_info['status']) +\
                   ' -ni=' + job_info['gpu_info'][0] +\
                   ' -gi=' + str(job_info['gpu_info'][1]) +\
                   ' -server=ncrs:5555' +\
                   ' -et=' + str(job_info['exec_tm']) +\
                   ' -wt=' + str(job_info['wait_tm'])

        exec_cmd = exec_cmd + ' -jid=' + str(job_info['id'])

        cmd = ssh + '"' + source + conda + cd + exec_cmd + '"'
        with open('cmd.txt', 'a+') as f:
            f.write(cmd)
            f.close()
        # An example here:
        # ssh ncrd "source .zshrc; conda activate stable;
        # source ~/configure/server1; cd models/official/resnet;
        # python cifar10_main.py --model_dir=/home/junda/large_bs"

        # create a new thread to execute the job
        new_job_thread = threading.Thread(target=self.exec_node, args=(cmd, ))
        new_job_thread.start()

    @staticmethod
    def exec_node(cmd):
        os.system(cmd)

    @staticmethod
    def execute_inception(job_info):
        pass

    @staticmethod
    def execute_vgg(job_info):
        pass
