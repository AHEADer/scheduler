# This file is to execute deep learning jobs
import os
import sys
from scheduler import Scheduler
from queue import Queue


class Executor:
    def __init__(self, a_scheduler, daemon_server):
        self.execute_queue = Queue()

    def execute(self, job, params, machine):
        if job == 'resnet':
            self.execute_resnet(params, machine)
        elif job == 'inception':
            self.execute_inception(params, machine)
        elif job == 'vgg':
            self.execute_vgg(params, machine)

    @staticmethod
    def execute_resnet(params, machine):
        # First ssh to that machine
        # Then execute code with hyperparameters
        # Note daemon process that a new job is running
        pass

    @staticmethod
    def execute_inception(params, machine):
        pass

    @staticmethod
    def execute_vgg(params, machine):
        pass
