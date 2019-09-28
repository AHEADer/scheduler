class Job:
    id = ''
    model = ''
    ep_tm = 0
    lock = True
    gpu_util = 0
    gpu_num = 0
    status = 'normal'
    grow_gpus = []
    grow_gpu_num = 0
    grow_node = ''
    node = ''
    gpus_loc = {}
    address = ''

    def dict_store(self, job_info):
        self.id = job_info['id']
        self.model = job_info['model']
        self.gpus_loc = job_info['gpus_loc']
        self.address = job_info['address']
        self.node = next(iter(self.gpus_loc))
        self.gpu_num = len(self.gpus_loc[self.node])


if __name__ == '__main__':
    job = Job()
    print(job.id)
