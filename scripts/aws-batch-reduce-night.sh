#!/bin/bash

mkdir -p /mnt/efs/scratch/RawData
cd /mnt/efs/scratch/RawData

# sync just the one night
aws s3 sync s3://projectdragonfly-archive/RawData/ . --exclude "*" --include "*$1*" --force-glacier-transfer

cd /

mkdir -p /mnt/efs/scratch/logs

/app/scripts/reduce-night -c /app/configs/awsbatch.yml --nproc 2 --log-fn /mnt/efs/scratch/logs/$1.log --date $1
