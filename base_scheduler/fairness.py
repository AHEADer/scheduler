import time
import copy
from utils import *
from job import *
from executor import Executor


class FairScheduler:
    def __init__(self, cluster_nodes, gpu_per_nodes, daemon=None):
        self.E = Executor()
        self.daemon = daemon
        self.init_job_queue = queue.Queue()
        self.resources = build_resource_dict(cluster_nodes, gpu_per_nodes)
        self.running_info = queue.Queue()
        self.running_jobs = {}
        self.growing_jobs = []
        self.shrinking_jobs = []
        self.msg_handler = threading.Thread(target=self._msg_handle, args=())
        self.msg_handler.start()
        self.occupy = 0

    def _msg_handle(self):
        while True:
            if self.occupy == 0:
                if not self.running_info.empty():
                    info = self.running_info.get()
                    info['gpus_loc'] = {'localhost': [0, 1, 2, 3, 4, 5, 6, 7]}
                    self.occupy = 1
                    self.E.exec(self.generate_new_job_by_info(info))

    @staticmethod
    def generate_new_job_by_info(info):
        new_job = Job()
        new_job.dict_store(info)
        return new_job

    def queue_empty(self):
        if not self.running_info.empty():
            return False
        else:
            return True

    def re_enqueue(self, info):
        self.running_info.put(info)