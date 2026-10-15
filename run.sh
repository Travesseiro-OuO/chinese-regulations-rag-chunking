#!/bin/sh
set -eu
cd /opt/rag-chunk-study-20261010
mkdir -p results
chown 10001:10001 results
docker run -d --name rag-chunk-experiment --network rag-chunk-internal --read-only --cap-drop ALL --security-opt no-new-privileges --memory 768m --memory-swap 1024m --cpus 1 --pids-limit 96 --tmpfs /tmp:rw,noexec,nosuid,size=64m -v "$PWD/src:/workspace/src:ro" -v "$PWD/data:/workspace/data:ro" -v "$PWD/models:/workspace/models:ro" -v "$PWD/results:/workspace/results:rw" rag-chunk-study:20261010 python src/experiment.py all
