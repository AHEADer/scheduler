import random
import time
import numpy as np
from pprint import pprint


class Generator:
    def __init__(self, a_scheduler):
        self.jobs_list = []
        self.scheduler = a_scheduler

    def generate(self, job_num):
        # batch size is discrete
        batch_size = [32, 64, 128, 256, 512, 1024, 2048, 4096]
        # learning rate is continuous
        for i in range(job_num):
            lr = round(random.uniform(1, 0.01), 2)
            tm = time.time() + random.uniform(2, 30)
            bs = batch_size[random.randrange(0,8)]
            # TODO adjust/tune this number
            epoch = 100
            job = {
                'model': 'resnet',
                'hyparams': [lr, bs, epoch],
                'join_tm': tm,
                'abnormal': False,
                'init_f': None,
                'r_tm': np.Inf,
                'join2_tm': 0
            }
            self.jobs_list.append([job, tm])
        self.jobs_list = sorted(self.jobs_list, key=lambda ins: ins[1])
            # pprint(self.jobs_list)

    def begin(self):
        print('generator begins!')
        last_tm = time.time()
        for each in self.jobs_list:
            remain = each[1]-last_tm
            print('next job will come in ' + str(remain) + ' sec')
            last_tm = each[1]
            if remain > 0:
                time.sleep(remain)
            self.scheduler.init_enqueue(each[0])
