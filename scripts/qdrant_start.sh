#!/bin/bash

docker run --ulimit nofile=65536:65536 --rm -d --name qdrant --network ies-network -v $HOME/innecs-local/qdrant_storage:/qdrant/storage:z -v /vol1/:/vol1:z -v $(pwd)/config/:/qdrant/config -p 6333:6333 -p 6334:6334 -e http_proxy="" qdrant/qdrant
echo "Open http://localhost:6333/dashboard#/collections to see all available collections."
