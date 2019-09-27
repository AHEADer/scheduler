import os


class Executor:
    def __init__(self):
        return

    def exec_resnet(self, job):
        ssh = self.ssh_cmd(job.node)
        combine = ''
        final_cmd = ssh + '"' + combine + '"'
        os.system(final_cmd)

    @staticmethod
    def gpu_cmd(gpu_list):
        gpus = ','.join(str(i) for i in gpu_list)
        cmd = 'env CUDA_VISIBLE_DEVICES='+gpus
        return cmd

    @staticmethod
    def combine_cmd(*cmds):
        cmd = ' '.join(_ for _ in cmds)
        return cmd

    @staticmethod
    def ssh_cmd(node):
        return 'ssh ' + node

