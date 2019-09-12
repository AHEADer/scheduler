# scheduler

job_info struct:
{

model: which model this job uses, string

hyparams: hyparameters for training, list or dict

join_tm: when job is submitted, float

abnormal: is there any anomaly, boolean

init_f: data retrived after initial training is finished, list

r_tm: predition remaining time

join2_tm: the second time when job finish its intial step and enter the queue

wait_tm: waiting time of this job 
}

scheduler.py：老的scheduler, 根据预测的时间进行优先级调度（静态）
new_scheduler.py: 新的动态scheduler

VGG training command:
env CUDA_VISIBLE_DEVICES=1,2,3,4,5 python vgg_cifar.py --id 1 --train --port 2001 
--data_path /home/ubuntu/xxx --saved_model /home/ubuntu/yyy/model.cpt 
--save_path /home/ubuntu/zzz --lr 0.1 --bsize 32 --keep_prob 0.5 --maxepoch 80
--gpus 1,2,3,4,5