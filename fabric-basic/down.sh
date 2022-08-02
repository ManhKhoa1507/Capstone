#!/bin/sh   
docker-compose down -v      
#docker container stop $(docker container ls -aq)
#docker container rm $(docker container ls -aq)                                
docker volume  prune 