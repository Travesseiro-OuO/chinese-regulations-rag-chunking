#!/bin/sh
set -eu
cd /opt/rag-chunk-study-20261010
docker network inspect rag-chunk-internal >/dev/null 2>&1 || docker network create --internal rag-chunk-internal
docker network inspect rag-chunk-egress >/dev/null 2>&1 || docker network create rag-chunk-egress
docker create --name rag-chunk-gateway --network rag-chunk-internal --network-alias api-relay --read-only --cap-drop ALL --security-opt no-new-privileges --memory 128m --cpus .25 --pids-limit 64 --tmpfs /tmp:rw,noexec,nosuid,size=16m -v "$PWD/src/gateway.py:/workspace/gateway.py:ro" -v "$PWD/secrets/deepseek_key:/run/secrets/deepseek_key:ro" rag-chunk-study:20261010 python gateway.py
docker network connect rag-chunk-egress rag-chunk-gateway
docker start rag-chunk-gateway
docker run --rm --network rag-chunk-internal --read-only --cap-drop ALL --security-opt no-new-privileges --memory 96m --cpus .25 rag-chunk-study:20261010 python -c 'import requests,time; time.sleep(2); r=requests.post("http://api-relay:8080/chat/completions",json={"model":"deepseek-chat","messages":[{"role":"user","content":"仅回复OK"}],"max_tokens":8,"temperature":0},timeout=90); d=r.json(); print("status",r.status_code,"model",d.get("model"),"reply",d.get("choices",[{}])[0].get("message",{}).get("content"),"error_type",d.get("error",{}).get("type"))'
