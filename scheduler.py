import threading
import queue
import time
from estimator import Estimator
# First try for 5 epochs and then analysis
# Predict the converged epoch, use patience strategy
# Training Time = epoch * time per epoch
# Base on the time use priority queue
# priority equals to (waiting time)*w1 + (train time)???
# compute fairness and total time

# Use message-based design


class Scheduler:
    def __init__(self, node_list, gpu_per_node, job_executor):
        self.job_queue = queue.Queue()
        self.init_job_queue = queue.Queue()
        self.cluster_resource = self.cluster_init(node_list, gpu_per_node)
        self.gpu_per_node = gpu_per_node
        self.current_wait_job = None
        self.job_executor = job_executor
        self.unpredicted_job = queue.Queue()
        self.msg_handler = threading.Thread(target=self._msg_handle, args=())
        self.msg_handler.start()

    def _msg_handle(self):
        while True:
            gpu_info = self.allocate_gpu()
            # see if there is a job get out of queue but waiting for schedule
            if gpu_info is None:
                # print('GPU is full')
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
                        self.execute_job(job_info, gpu_info)
                        pass
                else:
                    job_info = self.init_job_queue.get()
                    print('new job coming!')
                    # New job comes, check if there's available gpu resource
                    self.occupy_gpu(gpu_info)
                    self.execute_job(job_info, gpu_info)
                    # No new job coming, sleep some time

            # There's priority: trial job > less time job/long waited job > long time job

    def execute_job(self, job_info, gpu_info):
        # job info from generator or init step
        wait_tm = time.time()
        if job_info['status'] == 'init':
            job_info['wait_tm'] = wait_tm - job_info['join_tm']
        elif job_info['status'] == 'train':
            job_info['wait_tm'] = wait_tm - job_info['join2_tm'] + job_info['wait_tm']
        job_info['loc'] = '/tmp/' + job_info['model'] + job_info['id']
        job_info['gpu_info'] = gpu_info
        job_info['exec_tm'] = time.time()
        self.job_executor.execute(job_info)

    @staticmethod
    def cluster_init(node_list, gpu_per_node):
        cluster_info = {}
        for each in node_list:
            cluster_info[each] = [1 for i in range(gpu_per_node)]
        return cluster_info

    def allocate_gpu(self):
        for key in self.cluster_resource.keys():
            for i in range(self.gpu_per_node):
                if self.cluster_resource[key][i] == 1:
                    return key, i
        return None

    def occupy_gpu(self, gpu_info):
        self.cluster_resource[gpu_info[0]][gpu_info[1]] = 0

    def release_gpu(self, gpu_info):
        self.cluster_resource[gpu_info[0]][gpu_info[1]] = 1

    def priority(self, job_info):
        # basic:

        pass

    def init_enqueue(self, job_info):
        self.init_job_queue.put(job_info)

    def job_enqueue(self, job_info):
        # record current time
        cur_time = time.time()
        # get predict time
        # TODO change 5 to a parameter for scheduler
        tm_per_epoch = (job_info['join_tm'] - cur_time)/5
        est = Estimator()
        epoch = est.resnet_predict(job_info['init_f'], 0.01)
        if epoch == -1:
            # abnormal training, unpredictable！
            self.unpredicted_job.put(job_info)
        else:
            remaining_tm = epoch * tm_per_epoch
            job_info['r_tm'] = remaining_tm
            job_info['priority'] = 0
            job_info['join2_tm'] = time.time()
            self.job_queue.put(job_info)
        pass
