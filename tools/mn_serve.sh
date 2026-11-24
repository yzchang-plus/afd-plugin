#!/usr/bin/env bash
# 多节点弹性 AFD serve(基于 serve_2a2f_eager030b.sh)
# 前提: 两台机器已手动组成 Ray 集群(NODE1 是 head)
#   NODE1: ray start --head --port=6379 --num-gpus 8
#   NODE2: ray start --address='<NODE1_IP>:6379' --num-gpus 8
# 用法: 在 NODE1 上 cd scripts && nohup bash mn_serve.sh > /tmp/mn_serve.log 2>&1 &
set -eo pipefail
cd "$(dirname "$0")"

source /new_data/cyz/verify030b/venv030/bin/activate
_cann_env=""
for _c in /usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash \
          /usr/local/Ascend/ascend-toolkit/set_env.sh \
          /usr/local/Ascend/cann-9.1.0/set_env.sh; do
  if [ -f "$_c" ]; then _cann_env="$_c"; break; fi
done
[ -z "$_cann_env" ] && { echo "FATAL: CANN env not found" >&2; exit 1; }
source "$_cann_env"
[ -f /usr/local/Ascend/nnal/atb/set_env.sh ] && source /usr/local/Ascend/nnal/atb/set_env.sh

AFD_VENDOR=$(/new_data/cyz/verify030b/venv030/bin/python - << 'PYTHON'
from afd_plugin.compat.npu.ops import get_afd_cann_vendor_path
print(get_afd_cann_vendor_path())
PYTHON
)
test -d "$AFD_VENDOR" && export ASCEND_CUSTOM_OPP_PATH="${AFD_VENDOR}${ASCEND_CUSTOM_OPP_PATH:+:${ASCEND_CUSTOM_OPP_PATH}}" \
  && export LD_LIBRARY_PATH="$AFD_VENDOR/op_api/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

MODEL=/new_data/weight/DSV2LiteWeight

# 多节点: 本节点全卡可见(Ray 会把 actor 分布到两个节点)
# 不设 CARDS 限制; 第二个节点由 ray worker 自动管理
export ASCEND_RT_VISIBLE_DEVICES=${CARDS:-0,1,2,3,4,5,6,7}
export VLLM_USE_V1=1
export VLLM_PLUGINS=ascend,afd,afd_elastic
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export ASCEND_GLOBAL_LOG_LEVEL=3
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# 多节点关键: 不重启 Ray! 使用已建好的两节点集群
# (原脚本 53-54 行的 ray stop/ray start 已删)
echo "使用已有 Ray 集群(请确认 ray status 显示 2 nodes, 16 NPUs)"

exec vllm serve "$MODEL" \
    --host 127.0.0.1 \
    --port 8000 \
    --served-model-name dsv2lite \
    --seed 1024 \
    --max-model-len 4096 \
    --enforce-eager \
    --max-num-batched-tokens 16 \
    --max-num-seqs 16 \
    --gpu-memory-utilization 0.2 \
    --kv-cache-memory-bytes 2147483648 \
    --no-enable-prefix-caching \
    --data-parallel-size 2 \
    --data-parallel-size-local 2 \
    --data-parallel-backend ray \
    --distributed-executor-backend uni \
    --enable-expert-parallel \
    --enable-elastic-ep \
    --elastic-ep-max-dp-size 8 \
    --worker-cls afd_plugin.v1.worker.npu.AFDNPUAttentionWorker \
    --additional-config '{
      "afd": {
        "role": "attention",
        "connector": "CAMP2pAFDConnector",
        "host": "127.0.0.1",
        "port": 6241,
        "num_attention_ranks": 2,
        "num_ffn_ranks": 2,
        "compute_gate_on_attention": false
      },
      "afd_elastic": {
        "ffn_tensor_parallel_size": 1
      }
    }'
