class Job:
    id = ''
    ep_tm = 0
    lock = False
    gpu_util = 0
    gpu_num = 0
    status = 'norm'

    def dict_store(self, job_info):
        self.id = job_info['id']
        self.ep_tm = job_info['ep_tm']
        self.lock = job_info['lock']
        self.gpu_util = job_info['gpu_util']
        self.gpu_num = job_info['gpu_num']


if __name__ == '__main__':
    job = Job()
    print(job.id)
