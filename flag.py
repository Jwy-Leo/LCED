import tensorflow as tf

class flags:
    report_freq= 10
    train_maxiter=25000
    train_batch_size=1
    test_batch_size=1
    init_lr=0.0003
    lr_decay_factor=0.1
    decay_step0=15000
    decay_step1=20000
    num_residual_blocks=10
    weight_decay=0.0002
    padding_size=2
    numclass=10
    train=False



## The following flags are related to save paths, tensorboard outputs and screen outputs
