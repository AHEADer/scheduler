import os
import socket
import json


def binary_to_dict(the_binary):
    jsn = the_binary.decode('utf-8')
    d = json.loads(jsn)
    return d


def receiver():
    # open a server and wait for job report
    connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    connection.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    connection.bind(('0.0.0.0', 5555))
    connection.listen(10)
    while True:
        current_connection, address = connection.accept()
        data = current_connection.recv(2048)
        processed = binary_to_dict(data)
        print(processed['status'])


receiver()
