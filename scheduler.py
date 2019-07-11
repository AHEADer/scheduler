import threading
import queue
import time
from executor import *
# First try for 5 epochs and then analysis
# Predict the converged epoch, use patience strategy
# Training Time = epoch * time per epoch
# Base on the time use priority queue
# priority equals to (waiting time)*w1 + (train time)???
# compute fairness and total time

# Use message-based design

# job_info struct:
# {
# model
# hyparams
# join_time
# abnormal
# }


class Scheduler:
    def __init__(self, node_list, gpu_per_node, job_executor):
        self.job_queue = queue.Queue()
        self.init_job_queue = queue.Queue()
        self.cluster_resource = self.cluster_init(node_list, gpu_per_node)
        self.gpu_per_node = gpu_per_node
        self.current_wait_job = None
        self.job_executor = job_executor
        self.msg_handler = threading.Thread(target=self._msg_handle, args=())

    def _msg_handle(self):
        while True:
            gpu_info = self.allocate_gpu()
            # see if there is a job get out of queue but waiting for schedule
            if gpu_info is None:
                # TODO: set sleep time be the shortest remaining time of running jobs
                time.sleep(10)
                continue
            else:
                if self.init_job_queue.empty():
                    # no new job's coming, check jobs that finish initial step
                    if self.job_queue.empty():
                        # no jobs done nothing
                        pass
                    else:
                        job_info = self.job_queue.get()
                        self.occupy_gpu(gpu_info)
                        self.job_executor.execute(gpu_info, job_info)
                        pass
                else:
                    job_info = self.init_job_queue.get()
                    # New job comes, check if there's available gpu resource
                    self.occupy_gpu(gpu_info)
                    self.job_executor.execute(gpu_info, job_info)
                    # No new job coming, sleep some time
                    continue
            # There's priority: trial job > less time job/long waited job > long time job

    @staticmethod
    def cluster_init(node_list, gpu_per_node):
        cluster_info = {}
        for each in node_list:
            cluster_info[each] = [1 for i in range(gpu_per_node)]
        return cluster_info

    def allocate_gpu(self):
        for key in self.cluster_resource.keys():
            for i in range(self.gpu_per_node):
                if self.cluster_resource[key][1] == 1:
                    return key, i
        return None

    def occupy_gpu(self, gpu_info):
        self.cluster_resource[gpu_info[0]][gpu_info[1]] = 0

    def release_gpu(self, gpu_info):
        self.cluster_resource[gpu_info[0]][gpu_info[1]] = 1

    def priority(self, job_info):
        pass

    def job_enqueue(self, job_info):
        # record current time
        cur_time = time.time()
        # get predict time
        # TODO change 5 to a parameter for scheduler
        tm_per_epoch = (job_info['join_time'] - cur_time)/5
        # epoch =

        pass
