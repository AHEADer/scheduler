# This daemon process monitor running jobs and invoke scheduler once a job is done and release resource
import threading
import socket
import json
import time
from scheduler import Scheduler
from executor import Executor


def binary_to_dict(the_binary):
    jsn = the_binary.decode('utf-8')
    d = json.loads(jsn)
    return d


class Monitor:
    def __init__(self, a_scheduler, node_list, gpu_per_num):
        self.node_list = node_list
        self.a_scheduler = a_scheduler
        self.gpu_per_node = gpu_per_num
        self.msg_handler = threading.Thread(target=self._receiver, args=())
        self.msg_handler.start()

    def _receiver(self):
        # open a server and wait for job report
        connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        connection.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        connection.bind(('0.0.0.0', 5555))
        connection.listen(10)
        while True:
            current_connection, address = connection.accept()
            data = current_connection.recv(2048)

            job_return_info = binary_to_dict(data)
            # first release GPU resources
            self.a_scheduler.release_gpu(job_return_info['gpu_info'])
            if job_return_info['status'] == 'init':
                wait_tm = job_return_info['wait_tm']
                cpt_tm = job_return_info['cpt_tm']
                features = job_return_info['init_f']
                learning_rate = features[0]
                batch_size = 2**features[1]

                job_info = dict(model='resnet', init_f=features, wait_tm=wait_tm, cpt_tm=cpt_tm)
                job_info['id'] = job_return_info['id']
                job_info['hyparams'] = [learning_rate, batch_size]
                job_info['abnormal'] = 0
                # TODO make 5 be a parameter
                job_info['tm_per_ep'] = (float(job_return_info['cpt_tm']) - float(job_return_info['exec_tm']))/5
                job_info['status'] = 'train'

                # Then pass job to the queue, scheduler will do the fitting
                self.a_scheduler.job_enqueue(job_info)
            else:
                # print('one job complete')
                print(job_return_info)
                with open('complete.txt', 'a+', newline='\n') as f:
                    f.write(str(job_return_info))
                    f.close()
                # This job is completed, can measure its performance
                # TODO measure performance of a job


if __name__ == '__main__':
    node_list = ['ncrd']
    E = Executor()
    S = Scheduler(node_list, 2, E)
    M = Monitor(S, node_list, 3)





