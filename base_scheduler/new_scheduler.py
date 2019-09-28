import threading
import queue

import time
import heapq
import copy
from utils import *
from job import *

# from estimator import Estimator
# from job import Job

# First try for 5 epochs and then analysis
# Predict the converged epoch, use patience strategy
# Training Time = epoch * time per epoch
# Base on the time use priority queue
# priority equals to (waiting time)*w1 + (train time)???
# compute fairness and total time

# Use message-based design


class Scheduler:
    def __init__(self, cluster_nodes, gpu_per_nodes, daemon=None):
        # self.job_queue = []
        # self.gpus = [0, 0, 0, 0, 0, 0, 0, 0]
        # self.cluster_resource = self.cluster_init(node_list, gpu_per_node)
        # self.gpu_per_node = gpu_per_node
        # self.current_wait_job = None
        # self.job_executor = job_executor
        # self.unpredicted_job = queue.Queue()
        # self.estimator = Estimator()
        self.daemon = daemon
        self.init_job_queue = queue.Queue()
        self.resources = self._build_resource_dict(cluster_nodes, gpu_per_nodes)
        self.running_info = queue.Queue()
        self.running_jobs = {}
        self.growing_jobs = []
        self.shrinking_jobs = {}
        self.msg_handler = threading.Thread(target=self._msg_handle, args=())
        self.msg_handler.start()

    def _msg_handle(self):
        while True:
            while not self.running_info.empty():
                # TODO: below
                # Detect if there are available GPUs
                if self.allocate_gpu():
                    info = self.running_info.get()
                    self.update_running_info(info)
                else:
                    # currently no gpu can be allocated
                    time.sleep(10)

            if self.init_job_queue.empty():
                self.introspect()

    def set_daemon(self, daemon):
        self.daemon = daemon

    @staticmethod
    def _build_resource_dict(cluster_nodes, gpu_per_nodes):
        resource_dict = {}
        gpus = [0] * gpu_per_nodes
        for node in cluster_nodes:
            resource_dict[node] = copy.deepcopy(gpus)
        return resource_dict

    def check_free_gpu(self):
        free_dict = {}
        for node in self.resources.keys():
            node_list = []
            number = 0
            for gpu in self.resources[node]:
                if gpu == 0:
                    node_list.append(number)
                number += 1
            if len(node_list) > 0:
                free_dict[node] = node_list
        return free_dict

    def introspect(self):
        """ When there is no job in queue, try to grow-shrink current running jobs.
        Args:
        Return:
        """
        print('introspect start')
        jobs = self.get_running_jobs()
        if len(jobs) == 0:
            return

        available_nodes = self.check_free_gpu()
        if len(available_nodes) == 0:
            return

        single_node_max = max([len(available_nodes[l]) for l in available_nodes.keys()])

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
            if picked_job.gpu_num <= single_node_max:
                # allocate GPU to this job
                # TODO need to judge whether this job's gpu and allocated gpu in same node
                gpus_loc = self.get_gpus(picked_job.gpu_num, 'g')
                if gpus_loc[0] is None:
                    print('Error when allocating GPUs...')
                self.gpu_grow(picked_job, gpus_loc)
                available_nodes = self.check_free_gpu()
                single_node_max = max([len(available_nodes[l]) for l in available_nodes.keys()])
                # TODO lock jobs that assigned gpu
            else:
                continue

    def get_gpus(self, gpu_num, type):
        # type='g' growing, need to wait for checkpoint while new jobs might come
        # type ='n' the new job requires for GPUs and it will run immediately after getting the resource
        for node in self.resources.keys():
            node_list = []
            number = 0
            for gpu in self.resources[node]:
                if gpu == 0:
                    node_list.append(number)
            if len(node_list) >= gpu_num:
                for i in range(gpu_num):
                    if type == 'g':
                        self.resources[node][node_list[i]] = -1
                    elif type == 'n':
                        self.resources[node][node_list[i]] = 1
                return node, node_list[:gpu_num]
        return None, []

    def gpu_grow(self, job, gpu_loc):
        num_gpu = job.gpu_num
        node = gpu_loc[0]
        gpus = gpu_loc[1]
        job.gpu_num += len(gpus)
        if node in job.gpus_loc.keys():
            job.gpus_loc[node] = job.gpus_loc[node] + gpus
        else:
            job.gpus_loc[node] = gpus
        # Notify this job
        msg = {'type': 'g',
               'node': node,
               'gpus': gpus
               }
        send_msg(job.address, msg)
        job.status = 'growing'
        return 0

    def grow_ack(self, info):
        # make gpus occupied
        for key in info['gpus_loc']:
            for each in info['gpus_loc'][key]:
                self.resources[key][each] = 1
        self.growing_jobs.append(info['id'])
        return

    def gpu_shrink(self, job):
        shrink_gpu_num = job.gpu_num/2

        # TODO No multi node version
        for n in job.gpus_loc.keys():
            if len(job.gpus_loc[n]) >= shrink_gpu_num:
                node = n
                gpus = job.gpus_loc[n][:shrink_gpu_num]
                job.gpus_loc[n] = job.gpus_loc[n][shrink_gpu_num:]
                break
        msg = {'type': 's',
               'node': node,
               'gpus': gpus
               }
        send_msg(job.address, msg)
        job.status = 'shrinking'
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
            new_job = self.generate_new_job_by_info(info)
            self.running_jobs[info['id']] = new_job


    @staticmethod
    def generate_new_job_by_info(info):
        new_job = Job()
        new_job.dict_store(info)
        return new_job

    def allocate_gpu(self):
        free_dict = self.check_free_gpu()
        if free_dict != {}:
            return True
        # check growing gpu and recall them
        # TODO recall growing strategy is needed
        if self.growing_jobs == {}:
            # no growing jobs, which means cannot recall growing
            # then we need to find jobs to shrink
            # find jobs that is unlocked and have more than one GPU
            job_sk_list = []
            for job in self.running_jobs.keys():
                if job.lock is True or job.gpu_num == 1:
                    continue
                else:
                    job_sk_list.append(job)
            if len(job_sk_list) == 0:
                return False
            else:
                job_sk_list = sorted(job_sk_list, key=lambda item: item.gpu_num)
                # notify to shrink but still tell new jobs that cannot run now
                self.gpu_shrink(job_sk_list[0])
                return False
        else:
            grow_list = []
            for job_id in self.growing_jobs:
                grow_list.append(self.running_jobs[job_id])
            grow_list = sorted(grow_list, key=lambda item: item.grow_gpu_num)
            self.recall_grow(grow_list[0])
            return False

    def receive_running_info(self, info):
        self.running_info.put(info)

    def recall_grow(self, job):

        return

    def recall_shrink(self):
        return


