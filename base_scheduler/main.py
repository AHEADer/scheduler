from new_scheduler import Scheduler
from msg_daemon import Daemon
from params import *

if __name__ == '__main__':
    S = Scheduler(CLUSTER_NODES, GPU_PER_NODE)
    D = Daemon(S)
    S.set_daemon(D)
    print('haha')
