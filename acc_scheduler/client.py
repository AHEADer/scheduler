import socket
import json
import os


def dict_to_binary(the_dict):
    message = json.dumps(the_dict)
    print(str)
    return message.encode()


def binary_to_dict(the_binary):
    jsn = the_binary.decode('utf-8')
    d = json.loads(jsn)
    return d


# Create a TCP/IP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Connect the socket to the port where the server is listening
server_address = ('localhost', 5555)
print('connecting to {} port {}'.format(*server_address))
sock.connect(server_address)

try:
    # Send data
    message = {'status': 'init',
               'gpu_info': ('ncrb', 2),
               'loc': '/tmp/test',
               'init_f': 'tmp'}

    sock.sendall(dict_to_binary(message))

finally:
    print('closing socket')
    sock.close()