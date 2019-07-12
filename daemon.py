# This daemon process monitor running jobs and invoke scheduler once a job is done and release resource
import threading
import socket
import json


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

                # Then pass job to the queue, scheduler will do the fitting
                self.a_scheduler.job_enqueue()
            else:
                # This job is completed, can measure its performance
                # TODO measure performance of a job
                pass


if __name__ == '__main__':
    M = Monitor(None, ['ncra', 'ncrb', 'ncrc', 'ncrd'], 3)





