import time
import copy
from utils import *
from job import *
from executor import Executor


class FIFOScheduler:
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

    @staticmethod
    def _build_resource_dict(cluster_nodes, gpu_per_nodes):
        resource_dict = {}
        gpus = [0] * gpu_per_nodes
        for node in cluster_nodes:
            resource_dict[node] = copy.deepcopy(gpus)
        return resource_dict

    def _msg_handle(self):
        while True:
            free_gpus = self.return_free_gpus()['localhost']
            if len(free_gpus) >= 2:
                if not self.running_info.empty():
                    gpus = free_gpus[:2]
                    info = self.running_info.get()
                    if info['status'] == 'e':
                        del self.running_jobs[info['id']]
                    elif info['status'] == 'n':
                        info['gpus_loc'] = {'localhost': gpus}
                        new_job = self.generate_new_job_by_info(info)
                        self.running_jobs[info['id']] = new_job
                        self.E.exec(new_job)
                        for each in gpus:
                            self.resources['localhost'][each] = 0
                time.sleep(1)

    def release_gpu(self, job):
        for node in job.gpus_loc.keys():
            for each in job.gpus_loc[node]:
                self.resources[node][each] = 0

    @staticmethod
    def generate_new_job_by_info(info):
        new_job = Job()
        new_job.dict_store(info)
        return new_job

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

    def end(self, info):
        self.release_gpu(self.running_jobs[info['id']])
        del self.running_jobs[info['id']]

    def unlock(self, info):
        self.running_jobs[info['id']].lock = False
        self.running_jobs[info['id']].ep_tm = info['ep_tm']
