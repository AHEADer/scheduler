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