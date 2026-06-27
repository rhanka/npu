#!/usr/bin/env bash
# LANCE UN MODÈLE SUR LE NPU (Strix Halo XDNA2) via FastFlowLM — depuis les .deb
# extraits localement, SANS install système. Le seul besoin de root : lever la
# limite memlock (hard=8MB système) que FLM doit dépasser pour mapper le NPU.
#
# USAGE :  sudo bash run-npu.sh validate
#          sudo bash run-npu.sh run llama3.2:1b
#          sudo bash run-npu.sh run qwen2.5:1.5b
set -euo pipefail

DIR=/home/antoinefa/kog/npu-flm
USER_NAME=antoinefa
LIBS="$DIR/root/usr/lib/x86_64-linux-gnu:$DIR/root/opt/xilinx/xrt/lib:$DIR/root/opt/fastflowlm/lib"
FLM="$DIR/root/opt/fastflowlm/bin/flm"
CFG="$DIR/root/opt/fastflowlm/share/flm/model_list.json"

# prlimit lève memlock pour ce process; runuser redescend vers l'utilisateur
# (pour que les modèles HuggingFace s'écrivent dans son home, pas /root).
exec prlimit --memlock=unlimited:unlimited -- \
  runuser -u "$USER_NAME" -- \
  env LD_LIBRARY_PATH="$LIBS" \
      FLM_CONFIG_PATH="$CFG" \
      HOME="/home/$USER_NAME" \
  "$FLM" "$@"
