#!/bin/bash

# example.yaml
#name: example
#setup: |
#  echo "Run any setup commands here"
#  pip install cowsay
#run: |
#  echo "Hello Stranger!"
#  cowsay "Moo!"

sky launch 01_hello_sky/example.yaml
sky launch 01_hello_sky/example.yaml --cloud gcp
sky status
sky check
sky start | stop | down <cluster-name>

docker run --rm -p 8888:8888 \
-it public.ecr.aws/a9w6z7w5/skypilot-tutorial:latest
