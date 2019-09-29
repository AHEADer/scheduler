from new_scheduler import Scheduler
from msg_daemon import Daemon
from params import *
from generator import Generator

if __name__ == '__main__':
    S = Scheduler(CLUSTER_NODES, GPU_PER_NODE)
    G = Generator(S)
    D = Daemon(S)
    S.set_daemon(D)
    # Begin Generating
    G.generate(10, 10)
    G.begin()
    print('haha')
