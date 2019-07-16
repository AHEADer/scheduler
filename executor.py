# This file is to execute deep learning jobs
import os
import sys
from scheduler import Scheduler
from queue import Queue


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

    @staticmethod
    def execute_resnet(job_info):
        print(job_info)
        node = job_info['gpu_info'][0]
        gpu = job_info['gpu_info'][1]
        ssh = 'ssh ' + node
        gpu_pre = 'env CUDA_VISIBLE_DEVICES=' + str(gpu)
        conda = 'conda activate stable'
        source = 'source .zshrc; source ~/configure/server1'
        exec_cmd = 'python cifar10_main.py --md=' + job_info['loc'] +\
                   ' -lr=' + job_info['hyparams'][0] + ' -bs='+job_info['hyparams'][1] +\
                   ' -te=' + job_info['hyparams'][2]

        # An example here:
        # ssh ncrd "source .zshrc; conda activate stable;
        # source ~/configure/server1; cd models/official/resnet;
        # python cifar10_main.py --model_dir=/home/junda/large_bs"

        # First ssh to that machine
        # Then execute code with hyperparameters
        # Note daemon process that a new job is running
        pass

    @staticmethod
    def execute_inception(job_info):
        pass

    @staticmethod
    def execute_vgg(job_info):
        pass
