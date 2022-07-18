#!/bin/bash

mkdir -p /mnt/efs/scratch/logs

/app/scripts/reduce-target -c /app/configs/awsbatch.yml --nproc 2 --log-fn /mnt/efs/scratch/logs/$1.log --target $1
