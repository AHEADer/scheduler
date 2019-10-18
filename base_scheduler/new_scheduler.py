import threading
import queue

import time
import heapq
import copy
from utils import *
from job import *
from executor import Executor
import nvidia_smi
from logger import log_print

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
        self.E = Executor()
        self.daemon = daemon
        self.init_job_queue = queue.Queue()
        self.resources = self._build_resource_dict(cluster_nodes, gpu_per_nodes)
        self.running_info = queue.Queue()
        self.running_jobs = {}
        self.growing_jobs = []
        self.shrinking_jobs = []
        self.msg_handler = threading.Thread(target=self._msg_handle, args=())
        self.msg_handler.start()

    def _msg_handle(self):
        while True:
            log_print('while.txt', 'while is running')
            if not self.running_info.empty():
                # log_print('handler.txt', '--- print running queue ' + str(list(self.running_info.queue)))
                # TODO: below
                # Detect if there are available GPUs
                try:
                    if self.allocate_gpu():
                        info = self.running_info.get()
                        self.update_running_info(info)
                    else:
                        # currently no gpu can be allocated
                        log_print('scheduler.txt', '--- current no gpu allocated')
                        log_print('scheduler.txt', '----current GPU util: ' + str(self.resources))
                except:
                    log_print('while.txt', 'current gpu: ' + str(self.resources))
                    log_print('while.txt', '--- print running queue ' + str(list(self.running_info.queue)))
            elif self.init_job_queue.empty() and self.check_free_gpu() is True:
                self.introspect()
            time.sleep(5)

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
        for node in self.resources.keys():
            for gpu in self.resources[node]:
                if gpu == 0:
                    return True
        return False

    def return_free_gpus(self):
        free_dict = {}
        for node in self.resources.keys():
            node_list = []
            number = 0
            for gpu in self.resources[node]:
                if gpu == 0:
                    node_list.append(number)
                number += 1
            # if len(node_list) > 0:
            free_dict[node] = node_list
        return free_dict

    def introspect(self):
        """ When there is no job in queue, try to grow-shrink current running jobs.
        Args:
        Return:
        """
        # print('introspect start')
        jobs = self.get_running_jobs()
        if len(jobs) == 0:
            return

        available_nodes = self.return_free_gpus()
        if self.check_free_gpu() is False:
            return

        single_node_max = max([len(available_nodes[l]) for l in available_nodes.keys()])

        schedule_jobs = []
        for job in jobs:
            if job.id in self.growing_jobs:
                continue
            # if a job is locked, it is under speed test
            if not job.lock:
                # utilization
                if job.gpu_num == 1:
                    if self.cal_gpu_util(job) > 0.5:
                        schedule_jobs.append(job.id)
                elif self.cal_gpu_util(job) * job.gpu_num > job.gpu_num - 1:
                    schedule_jobs.append(job.id)
            else:
                continue

        # sort job by running time per epoch
        schedule_jobs = sorted(schedule_jobs, key=lambda item: self.running_jobs[item].ep_tm)
        # if len(schedule_jobs) != 0:
        #    print('try to introspect jobs')
        for jid in schedule_jobs:
            if self.running_jobs[jid].gpu_num <= single_node_max:
                # allocate GPU to this job
                # TODO need to judge whether this job's gpu and allocated gpu in same node
                gpus_loc = self.get_gpus(self.running_jobs[jid].gpu_num, 'g')
                if gpus_loc[0] is None:
                    log_print('scheduler.txt', 'Error when allocating GPUs, job id: ' + jid)
                self.gpu_grow(self.running_jobs[jid], gpus_loc)
                available_nodes = self.return_free_gpus()
                if self.check_free_gpu() is False:
                    single_node_max = 0
                else:
                    single_node_max = max([len(available_nodes[l]) for l in available_nodes.keys()])
                # TODO lock jobs that assigned gpu
            else:
                continue

    @staticmethod
    def cal_gpu_util(job):
        ct = 0
        gpu = 0
        nvidia_smi.nvmlInit()
        for key in job.gpus_loc.keys():
            for i in job.gpus_loc[key]:
                ct += 1
                handle = nvidia_smi.nvmlDeviceGetHandleByIndex(i)
                res = nvidia_smi.nvmlDeviceGetUtilizationRates(handle)
                gpu += res.gpu
        if ct > 0:
            avg = gpu/ct
            return avg
        else:
            print('job no gpu')
            return 0

    def get_gpus(self, gpu_num, type):
        # type='g' growing, need to wait for checkpoint while new jobs might come
        # type ='n' the new job requires for GPUs and it will run immediately after getting the resource
        for node in self.resources.keys():
            node_list = []
            number = 0
            for gpu in self.resources[node]:
                if gpu == 0:
                    node_list.append(number)
                number += 1
            if len(node_list) >= gpu_num:
                for i in range(gpu_num):
                    if type == 'g':
                        self.resources[node][node_list[i]] = -1
                    elif type == 'n':
                        self.resources[node][node_list[i]] = 1
                return node, node_list[:gpu_num]
        return None, []

    def gpu_grow(self, job, gpu_loc):
        node = gpu_loc[0]
        gpus = gpu_loc[1]
        self.running_jobs[job.id].gpu_num += len(gpus)
        if node in job.gpus_loc.keys():
            self.running_jobs[job.id].gpus_loc[node] = job.gpus_loc[node] + gpus
        else:
            self.running_jobs[job.id].gpus_loc[node] = gpus
        # Notify this job
        msg = {'status': 'g',
               'node': node,
               'gpus': gpus
               }
        send_msg(job.address, msg)
        self.running_jobs[job.id].status = 'growing'
        self.running_jobs[job.id].lock = True
        self.growing_jobs.append(job.id)
        log_print('scheduler.txt', 'job ' + job.id + ' is growing')
        return 0

    def grow_ack(self, info):
        # make gpus occupied
        ct = 0
        for key in info['gpus_loc']:
            for each in info['gpus_loc'][key]:
                ct += 1
                self.resources[key][each] = 1

        self.running_jobs[info['id']].gpu_num = ct
        # speed test again
        self.running_jobs[info['id']].lock = True
        self.running_jobs[info['id']].status = 'n'
        self.running_jobs[info['id']].ep = info['ep']
        # sleep 1 seconds waiting for GPU release
        self.growing_jobs.remove(info['id'])
        time.sleep(1)
        log_print('scheduler.txt', 'job ' + info['id'] + ' has grown')
        self.E.exec(self.running_jobs[info['id']])

    def shrink_ack(self, info):
        ct = 0
        log_print('scheduler.txt', 'job ' + info['id'] + ' begin to shrink')
        for key in self.running_jobs[info['id']].gpus_loc.keys():
            for each in self.running_jobs[info['id']].gpus_loc[key]:
                if self.resources[key][each] == -2:
                    ct += 1
                    self.resources[key][each] = 0

        self.running_jobs[info['id']].gpu_num -= ct
        self.running_jobs[info['id']].gpus_loc = info['gpus_loc']
        # speed test again
        self.running_jobs[info['id']].lock = True
        self.running_jobs[info['id']].ep = info['ep']
        # sleep 1 seconds waiting for GPU release
        self.shrinking_jobs.remove(info['id'])
        self.running_jobs[info['id']].status = 'n'
        log_print('scheduler.txt', 'job ' + info['id'] + ' has shrunk')
        self.E.exec(self.running_jobs[info['id']])

    def gpu_shrink(self, job):
        log_print('scheduler.txt', '--- gpu shrink, job id: ' + job.id)
        shrink_gpu_num = int(job.gpu_num/2)
        # TODO hardcoded here
        node = 'localhost'
        gpus = []
        # TODO No multi node version
        log_print('scheduler.txt', '--- print job: ' + str(vars(job)))
        # below has error?
        for n in job.gpus_loc.keys():
            if len(job.gpus_loc[n]) >= shrink_gpu_num:
                # node = n
                gpus = job.gpus_loc[n][:shrink_gpu_num]
                # job.gpus_loc[n] = job.gpus_loc[n][shrink_gpu_num:]
                # break
        # above is OK
        for gpu in gpus:
            self.resources[node][gpu] = -2

        msg = {'status': 's',
               'node': node,
               'gpus': gpus
               }
        send_msg(job.address, msg)
        self.running_jobs[job.id].status = 'shrinking'
        self.running_jobs[job.id].lock = True
        self.shrinking_jobs.append(job.id)
        return 0

    def get_running_jobs(self):
        _ = []
        for key in self.running_jobs.keys():
            _.append(self.running_jobs[key])
        return _

    def update_running_info(self, info):
        log_print('scheduler.txt', 'update job: ' + str(info))
        if info['status'] == 'e':
            del self.running_jobs[info['id']]
        elif info['status'] == 'n':
            gpu_tu = self.get_gpus(1, 'n')
            info['gpus_loc'] = {gpu_tu[0]: gpu_tu[1]}
            new_job = self.generate_new_job_by_info(info)
            # get one GPU for it to run
            self.running_jobs[info['id']] = new_job
            # print('exec job')
            self.E.exec(new_job)

    def unlock(self, info):
        log_print('scheduler.txt', '----unlock job ' + info['id'])
        self.running_jobs[info['id']].lock = False
        self.running_jobs[info['id']].ep_tm = info['ep_tm']

    def end(self, info):
        log_print('scheduler.txt', '----end job ' + info['id'])
        time.sleep(1)
        self.release_gpu(self.running_jobs[info['id']])
        del self.running_jobs[info['id']]
        log_print('scheduler.txt', '----current GPU util: ' + str(self.resources))
        for each in self.running_jobs.keys():
            log_print('scheduler.txt', '----job log: ' + str(vars(self.running_jobs[each])))
        for each_id in self.shrinking_jobs:
            log_print('scheduler.txt', '----recall shrink job: ' + str(each_id))
            self.recall_shrink(self.running_jobs[each_id])
        log_print('scheduler.txt', '----recall shrink done')

    def release_gpu(self, job):
        for node in job.gpus_loc.keys():
            for each in job.gpus_loc[node]:
                self.resources[node][each] = 0

    @staticmethod
    def generate_new_job_by_info(info):
        new_job = Job()
        new_job.dict_store(info)
        return new_job

    def allocate_gpu(self):
        # free_dict = self.return_free_gpus()
        if self.check_free_gpu() is True:
            return True
        # check growing gpu and recall them
        # TODO recall growing strategy is needed
        if not self.growing_jobs:
            # no growing jobs, which means cannot recall growing
            # then we need to find jobs to shrink
            # find jobs that is unlocked and have more than one GPU
            job_sk_list = []
            for job_id in self.running_jobs.keys():
                if self.running_jobs[job_id].lock or self.running_jobs[job_id].gpu_num == 1 or self.running_jobs[job_id].status != 'n':
                    continue
                else:
                    job_sk_list.append(self.running_jobs[job_id])
            if len(job_sk_list) == 0:
                return False
            else:
                job_sk_list = sorted(job_sk_list, key=lambda item: self.cal_gpu_util(item))
                # notify to shrink but still tell new jobs that cannot run now
                self.gpu_shrink(job_sk_list[-1])
                return False
        else:
            grow_list = []
            for job_id in self.growing_jobs:
                grow_list.append(self.running_jobs[job_id])
            if len(grow_list) == 0:
                return False
            grow_list = sorted(grow_list, key=lambda item: self.cal_gpu_util(item))
            # recall one that gpu utilization is the lowest
            self.recall_grow(grow_list[-1])
            return False

    def receive_running_info(self, info):
        log_print('scheduler.txt', '--- receive a job ' + info['id'] + ' to queue')
        self.running_info.put(info)

    def recall_grow(self, job):
        log_print('scheduler.txt', '--- recall grow, job id: ' + job.id)
        gpus = []
        n = ''
        for node in job.gpus_loc.keys():
            for gpu in job.gpus_loc[node]:
                if self.resources[node][gpu] == -1:
                    self.resources[node][gpu] = 0
                    gpus.append(gpu)
                    n = node

        msg = {'status': 'rg',
               'node': n,
               'gpus': gpus
               }
        send_msg(job.address, msg)
        self.growing_jobs.remove(job.id)

    def recall_shrink(self, job):
        log_print('scheduler.txt', '--- recall shrink, job id: ' + job.id)
        gpus = []
        n = ''
        for node in job.gpus_loc.keys():
            n = node
            for gpu in job.gpus_loc[node]:
                gpus.append(gpu)
                if self.resources[node][gpu] == -2:
                    self.resources[node][gpu] = 1

        msg = {'status': 'rs',
               'node': n,
               'gpus': gpus
               }
        send_msg(job.address, msg)
        self.shrinking_jobs.remove(job.id)


