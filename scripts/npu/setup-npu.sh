#!/usr/bin/env bash
# SETUP NPU RYZEN AI (XDNA2 / RyzenAI-npu5) — XRT + amdxdna + Ryzen AI Software.
#
# Le kernel est déjà prêt (driver amdxdna chargé, firmware /lib/firmware/amdnpu/).
# Il manque le userspace : XRT (runtime) + VitisAI EP (compilateur ONNX→NPU).
#
# /!\ ÉTAPE GATÉE : le bundle ryzen_ai-1.7.1.tgz + xrt_plugin .deb ne sont
#     téléchargeables que depuis le portail compte AMD (login + EULA) :
#     https://ryzenai.docs.amd.com/en/latest/linux.html
#     Le module `voe` / VitisAI EP qui exécute l'ONNX sur le NPU y est inclus.
#
# /!\ AMD documente le runtime Linux comme INSTABLE (bug ouvert #341 :
#     VitisAI EP échoue à offloader sur le NPU sous Linux x86_64).
#
# USAGE : sudo bash scripts/setup-npu.sh [chemin-vers-ryzen_ai-1.7.1.tgz]
set -euo pipefail

XDNA_SRC=/tmp/xdna-driver
RYZEN_TGZ="${1:-}"

echo "══════════════════════════════════════════════"
echo " SETUP NPU RYZEN AI — XDNA2 / RyzenAI-npu5"
echo "══════════════════════════════════════════════"

# ── 1. VÉRIF KERNEL (déjà prêt normalement) ──────────────────────────────────
echo "[1/5] VÉRIFICATION KERNEL NPU…"
lsmod | grep -q amdxdna && echo "  ✅ driver amdxdna chargé" || {
  echo "  ❌ amdxdna non chargé — modprobe amdxdna"; modprobe amdxdna || true; }
ls /dev/accel0 >/dev/null 2>&1 && echo "  ✅ /dev/accel0 présent" || echo "  ❌ /dev/accel0 absent"
ls /lib/firmware/amdnpu/ >/dev/null 2>&1 && echo "  ✅ firmware présent" || echo "  ❌ firmware NPU absent"

# ── 2. BUILD XRT + AMDXDNA (PUBLIC, github.com/amd/xdna-driver) ───────────────
echo ""
echo "[2/5] BUILD XRT + PLUGIN AMDXDNA (depuis sources publiques)…"
if [[ ! -d "$XDNA_SRC" ]]; then
  git clone --recurse-submodules https://github.com/amd/xdna-driver.git "$XDNA_SRC"
fi
cd "$XDNA_SRC"
echo "  -- install build deps (amdxdna_deps.sh) --"
bash tools/amdxdna_deps.sh
echo "  -- build XRT base (submodule) --"
cd "$XDNA_SRC/xrt/build" && ./build.sh -npu -opt
echo "  -- install XRT base .deb --"
apt install --fix-broken -y "$XDNA_SRC"/xrt/build/Release/xrt_*-amd64-base.deb || true
echo "  -- build amdxdna plugin --"
cd "$XDNA_SRC/build" && ./build.sh -release && ./build.sh -package
apt install --fix-broken -y "$XDNA_SRC"/build/Release/xrt_plugin*-amdxdna.deb || true

# ── 3. ENV + VÉRIF NPU ───────────────────────────────────────────────────────
echo ""
echo "[3/5] VÉRIFICATION NPU VIA XRT…"
export LD_LIBRARY_PATH=/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}
[[ -f /opt/xilinx/xrt/setup.sh ]] && source /opt/xilinx/xrt/setup.sh
xrt-smi examine 2>&1 | head -20 || echo "  ⚠️ xrt-smi indisponible"

# ── 4. RYZEN AI SOFTWARE (GATÉ — VitisAI EP / voe) ───────────────────────────
echo ""
echo "[4/5] RYZEN AI SOFTWARE (VitisAI EP)…"
if [[ -n "$RYZEN_TGZ" && -f "$RYZEN_TGZ" ]]; then
  WORK=/tmp/ryzen_ai-1.7.1; mkdir -p "$WORK"; cp "$RYZEN_TGZ" "$WORK/"
  cd "$WORK" && tar -xzf "$(basename "$RYZEN_TGZ")"
  ./install_ryzen_ai.sh -a yes -p /home/antoinefa/kog/quant/.venv-npu
  echo "  ✅ Ryzen AI installé dans .venv-npu"
else
  echo "  ⏭️  bundle non fourni. Télécharge ryzen_ai-1.7.1.tgz depuis ton compte AMD :"
  echo "      https://ryzenai.docs.amd.com/en/latest/linux.html"
  echo "      puis relance : sudo bash scripts/setup-npu.sh /chemin/ryzen_ai-1.7.1.tgz"
fi

# ── 5. TEST INFÉRENCE NPU (modèle INT8) ──────────────────────────────────────
echo ""
echo "[5/5] PROCHAINE ÉTAPE — inférence NPU :"
echo "  python scripts/npu_infer.py  (VitisAI EP sur le modèle INT8 streaming)"
echo ""
echo "══════════════════════════════════════════════"
echo " NPU READY (modulo bundle AMD + bug #341 éventuel)"
echo "══════════════════════════════════════════════"
