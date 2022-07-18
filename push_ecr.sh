#!/bin/bash
aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin 935670445359.dkr.ecr.us-west-2.amazonaws.com
docker build -t dfreduce .
docker tag dfreduce:latest 935670445359.dkr.ecr.us-west-2.amazonaws.com/dfreduce:latest
docker push 935670445359.dkr.ecr.us-west-2.amazonaws.com/dfreduce:latest
