from scheduler import Scheduler
from generator import Generator
from executor import Executor
# First generate jobs
# Pass job to scheduler

if __name__ == '__main__':
    E = Executor()
    node_list = ['ncrd']
    S = Scheduler(node_list, 3, E)
    G = Generator(S)
    G.generate(13)
    G.begin()
