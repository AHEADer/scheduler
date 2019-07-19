import threading
import queue
import time
import heapq
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
        self.job_queue = []
        self.init_job_queue = queue.Queue()
        self.cluster_resource = self.cluster_init(node_list, gpu_per_node)
        self.gpu_per_node = gpu_per_node
        self.current_wait_job = None
        self.job_executor = job_executor
        self.unpredicted_job = queue.Queue()
        self.estimator = Estimator()
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
                    if len(self.job_queue) > 0:
                        # first update the priority queue
                        self.update_job_queue()
                        print('train job exec')
                        job_info = heapq.heappop(self.job_queue)[1]
                        self.occupy_gpu(gpu_info)
                        self.execute_job(job_info, gpu_info)
                else:
                    job_info = self.init_job_queue.get()
                    print('new job coming!')
                    # New job comes, check if there's available gpu resource
                    self.occupy_gpu(gpu_info)
                    self.execute_job(job_info, gpu_info)
                    # No new job coming, sleep some time

            # There's priority: trial job > less time job/long waited job > long time job

    def update_job_queue(self):
        current_tm = time.time()
        for i in range(len(self.job_queue)):
            self.job_queue[i][1]['wait_tm'] += current_tm - self.job_queue[i][1]['cpt_tm']
            self.job_queue[i][0] = self.priority(self.job_queue[i][1]['wait_tm'], self.job_queue[i][1]['r_tm'])

        heapq.heapify(self.job_queue)

    def execute_job(self, job_info, gpu_info):
        self.log_write(job_info)
        # job info from generator or init step
        wait_tm = time.time()
        if job_info['status'] == 'init':
            job_info['wait_tm'] = wait_tm - job_info['join_tm']
        elif job_info['status'] == 'train':
            job_info['wait_tm'] = wait_tm - job_info['join2_tm'] + job_info['wait_tm']
        job_info['loc'] = '/tmp/' + job_info['model'] + str(job_info['id'])
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
        self.cluster_resource[gpu_info[0]][int(gpu_info[1])] = 1

    @staticmethod
    def priority(wait_tm, r_tm):
        # basic: r_tm must be more important than wait_tm
        return r_tm/(wait_tm+5*60)

    def init_enqueue(self, job_info):
        self.init_job_queue.put(job_info)

    @staticmethod
    def log_write(job_info):
        with open('log.txt', 'a+', newline='\n') as f:
            f.write(str(job_info))
            f.close()

    def job_enqueue(self, job_info):
        # record current time
        cur_time = time.time()
        # get predict time
        # TODO change 5 to a parameter for scheduler
        tm_per_epoch = job_info['tm_per_ep']
        epoch = self.estimator.resnet_predict(job_info['init_f'], 0.01)
        if epoch == -1:
            # abnormal training, unpredictable！
            self.unpredicted_job.put(job_info)
        else:
            job_info['hyparams'].append(epoch)
            remaining_tm = epoch * tm_per_epoch
            job_info['r_tm'] = remaining_tm
            job_info['join2_tm'] = time.time()
            job_info['wait_tm'] = job_info['join2_tm'] - job_info['cpt_tm'] + job_info['wait_tm']
            priority = self.priority(job_info['wait_tm'], remaining_tm)
            job_info['priority'] = priority
            heapq.heappush(self.job_queue, [priority, job_info])
