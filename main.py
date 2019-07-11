from scheduler import Scheduler
from generator import Generator
from executor import Executor
# First generate jobs
# Pass job to scheduler

if __name__ == '__main__':
    E = Executor()
    node_list = ['ncra', 'ncrb', 'ncrc', 'ncrd']
    S = Scheduler(node_list, 2, E)
    G = Generator(S)
    G.generate(13)
    G.begin()
