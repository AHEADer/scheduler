class Job:
    ep = 0
    id = ''
    model = ''
    ep_tm = 0
    lock = True
    gpu_util = 0
    gpu_num = 0
    status = 'n'
    grow_gpus = []
    grow_gpu_num = 0
    grow_node = ''
    node = ''
    gpus_loc = {}
    address = ''

    def dict_store(self, job_info):
        self.status = job_info['status']
        self.id = job_info['id']
        self.model = job_info['model']
        self.gpus_loc = job_info['gpus_loc']
        self.address = job_info['address']
        self.node = next(iter(self.gpus_loc))
        self.gpu_num = len(self.gpus_loc[self.node])
        self.ep = job_info['ep']
        self.ep_tm = 0
        self.lock = True
        self.grow_gpus = []
        self.grow_gpu_num = 0
        self.grow_node = ''


if __name__ == '__main__':
    job = Job()
    info = {
        'id': '1',
        'model': 'resnet50',
        # n means normal here
        'status': 'n',
        'address': 'localhost:2345',
        'ep': 2,
        'gpus_loc': {'localhost': [2, 3]}
    }
    job.dict_store(info)
    print(vars(job))
