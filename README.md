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
}