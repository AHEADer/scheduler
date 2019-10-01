import random
import time
import numpy as np
from pprint import pprint


class Generator:
    def __init__(self, a_scheduler):
        self.models_list = ['vgg16', 'vgg19', 'inceptionv3', 'xception', 'resnet50', 'inceptionresnetv2',
                            'mobilenet', 'densenet121', 'densenet169', 'densenet201', 'nasnetlarge', 'nasnetmobile']
        self.jobs_list = []
        self.scheduler = a_scheduler

    def generate(self, job_num, time_span, epoch_span):
        # batch size is discrete
        batch_size = [32, 64, 128, 256, 512, 1024, 2048]
        start = 8888
        # learning rate is continuous
        for i in range(job_num):
            id = str(i+1)
            tm = time.time() + random.uniform(0, time_span)
            bs = batch_size[random.randrange(0, 7)]
            model = self.models_list[random.randrange(0, 12)]
            # TODO adjust/tune this number
            job = {
                'id': id,
                'model': model,
                # n means normal here
                'status': 'n',
                'address': 'localhost:' + str(start),
                'ep': random.randint(epoch_span[0], epoch_span[1])
            }
            start += 1
            self.jobs_list.append([job, tm])
        self.jobs_list = sorted(self.jobs_list, key=lambda ins: ins[1])

    def begin(self):
        print('generator begins!')
        last_tm = time.time()
        for each in self.jobs_list:
            remain = each[1]-last_tm
            print('next job will come in ' + str(remain) + ' sec')
            last_tm = each[1]
            if remain > 0:
                time.sleep(remain)
            self.scheduler.receive_running_info(each[0])


if __name__ == '__main__':
    G = Generator('haha')
    G.generate(10, 20)
    G.begin()
