from utils import binary_to_dict
from logger import log_print
import socket
import threading
import json


class Daemon:
    def __init__(self, scheduler=None):
        self.scheduler = scheduler
        self.msg_handler = threading.Thread(target=self.receive, args=())
        self.msg_handler.start()

    def set_scheduler(self, scheduler):
        self.scheduler = scheduler

    def receive(self):
        # open a server and wait for job report
        connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        connection.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        connection.bind(('localhost', 1080))
        connection.listen(50)
        while True:
            current_connection, address = connection.accept()
            data = current_connection.recv(2048)
            info = binary_to_dict(data)
            log_print('daemon.txt', 'receive a job, id = ' + info['id'])
            if info['status'] == 'g':
                self.receive_grow(info)
            elif info['status'] == 's':
                self.receive_shrink(info)
            elif info['status'] == 'e':
                self.receive_end(info)
            elif info['status'] == 'un':
                self.unlock(info)

    # scheduler part begin
    def ask_grow(self):

        return

    def ask_shrink(self):
        return

    def recall_grow(self):
        return

    def recall_shrink(self):
        return

    def migrate(self):
        return

    def merge(self):
        return
    # scheduler part end

    # job part begin
    def receive_grow(self, info):
        # print(info)
        self.scheduler.grow_ack(info)
        return

    def receive_recall_grow(self):
        return

    def receive_shrink(self, info):
        self.scheduler.shrink_ack(info)
        print(info)

    def receive_end(self, info):
        log_print('daemon.txt', 'end job id: ' + info['id'])
        self.scheduler.end(info)

    def unlock(self, info):
        log_print('daemon.txt', 'unlock job id: ' + info['id'])
        self.scheduler.unlock(info)
    # job part end
