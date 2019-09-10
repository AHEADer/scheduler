import threading
import queue
import time
import heapq
from estimator import Estimator
from job import Job
# First try for 5 epochs and then analysis
# Predict the converged epoch, use patience strategy
# Training Time = epoch * time per epoch
# Base on the time use priority queue
# priority equals to (waiting time)*w1 + (train time)???
# compute fairness and total time

# Use message-based design


class Scheduler:
    # def __init__(self, node_list, gpu_per_node, job_executor):
    def __init__(self):
        # self.job_queue = []
        self.init_job_queue = queue.Queue()
        self.gpus = [0, 0, 0, 0, 0, 0, 0, 0]
        # self.cluster_resource = self.cluster_init(node_list, gpu_per_node)
        # self.gpu_per_node = gpu_per_node
        # self.current_wait_job = None
        # self.job_executor = job_executor
        # self.unpredicted_job = queue.Queue()
        # self.estimator = Estimator()
        self.running_info = queue.Queue()
        self.running_jobs = {}
        self.msg_handler = threading.Thread(target=self._msg_handle, args=())
        self.msg_handler.start()

    def _msg_handle(self):
        while True:
            while not self.running_info.empty():
                info = self.running_info.get()
                self.update_running_info(info)
            if self.init_job_queue.empty():
                self.introspect()

    def check_free_gpu(self):
        num = len(self.gpus)
        _ = []
        for i in range(num):
            if self.gpus[i] == 0:
                _.append(i)
        return _

    def introspect(self):
        """ When there is no job in queue, try to grow-shrink current running jobs.
        Args:
        Return:
        """
        jobs = self.get_running_jobs()
        if len(jobs) == 0:
            return

        available_gpus = self.check_free_gpu()
        num_available_gpu = len(available_gpus)
        if num_available_gpu == 0:
            return

        schedule_jobs = []
        for job in jobs:
            # if a job is locked, it is under speed test
            if not job.lock:
                # utilization
                if job.gpu_util * job.gpu_num > job.gpu_num - 1:
                    schedule_jobs.append(job)
            else:
                continue

        # sort job by running time per epoch
        schedule_jobs = sorted(schedule_jobs, key=lambda item: item.ep_tm)
        for picked_job in schedule_jobs:
            if picked_job.gpu_num < num_available_gpu:
                num_available_gpu = self.gpu_grow(picked_job)
                # TODO lock jobs that assigned gpu
            else:
                continue

    def gpu_grow(self, job):
        num_gpu = job.gpu_num

        return 0

    def get_running_jobs(self):
        _ = []
        for key in self.running_jobs.keys():
            _.append(self.running_jobs[key])
        return _

    def update_running_info(self, info):
        if info['status'] == 'end':
            del self.running_jobs[info['id']]
        else:
            self.running_jobs[info['id']] = info

    def receive_running_info(self, info):
        self.running_info.put(info)

