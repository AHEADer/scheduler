# This daemon process monitor running jobs and invoke scheduler once a job is done and release resource
import threading
import socket

class Monitor:
    def __init__(self, a_scheduler, node_list, gpu_per_num):
        self.node_list = node_list
        self.gpu_per_node = gpu_per_num
        self.msg_handler = threading.Thread(target=self._loop_check, args=())
        self.msg_handler.start()

    def _loop_check(self):
        while True:
            continue
        pass
