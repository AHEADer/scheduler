import threading
import socket
import queue
import json


# receive/send message
class Server:
    def __init__(self, Q, port):
        if not isinstance(Q, queue.Queue):
            print('argument Q not queue')
        self.Q = Q
        self.server_daemon = threading.Thread(target=self.start(port,), args=())
        self.server_daemon.start()

    def start(self, port):
        connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        connection.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        connection.bind(('0.0.0.0', port))
        connection.listen(10)
        while True:
            current_connection, address = connection.accept()
            data = current_connection.recv(2048)
            self.Q.put(binary_to_dict(data))
            print(address + ' send a message')


def binary_to_dict(the_binary):
    jsn = the_binary.decode('utf-8')
    d = json.loads(jsn)
    return d


def dict_to_binary(the_dict):
    message = json.dumps(the_dict)
    return message.encode()


def send_msg(address, message):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    ip_address, port = address.split(':')
    server_address = (ip_address, int(port))
    sock.connect(server_address)
    try:
        sock.sendall(dict_to_binary(message))
    finally:
        sock.close()

