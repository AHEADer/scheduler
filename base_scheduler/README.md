### job status: 

g(growing)

l(lock) under speed test, at least run 100 steps/1 epochs

s(shrinking)

### scheduler strategy


### Job message to daemon monitor
1. 'id': Job id
2. 'status': What kind of response
    + 'g', the job has receive grow message and begin to grow.
    + 's', the job has receive shrink message and begin to shrink.1
    + 'e', the job ends.
    + 'un', the job has ended the speed test, unlock message
    (which means we can grow and shrink on this job)
    + 'n', the job is normal, which indicates the begin part
3. 'model': What kind of model this job uses
    + 'vgg16', 'vgg19', 'inceptionv3', 'xception', 'resnet50', 'inceptionresnetv2',
     'mobilenet', 'densenet121', 'densenet169', 'densenet201', 'nasnetlarge', 'nasnetmobile'
4. 'gpus_loc': A dict contains nodes and gpus, for example: {'0.0.0.0': [1,2,3,5]}
5.  