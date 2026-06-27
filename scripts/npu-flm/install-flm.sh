#!/usr/bin/env bash
# INSTALL FASTFLOWLM (NPU XDNA2 / Strix Halo) — Ubuntu 26.04, kernel 7.0+.
# Tout est déjà téléchargé (3 .deb publics, aucun gate AMD). Un seul sudo.
#
# Prérequis déjà OK sur cette machine : kernel 7.0.0, amdxdna chargé, NPU FW 1.1.2.65.
# USAGE : sudo bash install-flm.sh
set -euo pipefail
cd "$(dirname "$0")"

echo "══════════════════════════════════════════════"
echo " FASTFLOWLM — NPU Strix Halo (XDNA2)"
echo "══════════════════════════════════════════════"

# ── 1. libxrt (runtime XRT NPU) — build questing 25.10, compatible 26.04 ────
echo "[1/4] install libxrt2 + libxrt-npu2…"
apt install -y --allow-downgrades \
  ./libxrt2_2.21.75-1~questing1_amd64.deb \
  ./libxrt-npu2_2.21.75-1~questing1_amd64.deb

# ── 2. FastFlowLM (+ deps ffmpeg/boost depuis les dépôts Ubuntu) ─────────────
echo "[2/4] install FastFlowLM (deps auto)…"
apt install -y ./fastflowlm_ubuntu26.04.deb

# ── 3. memlock infinity (requis par FLM pour mapper le NPU) ──────────────────
echo "[3/4] memlock → infinity…"
cat > /etc/security/limits.d/flm.conf <<'EOF'
* hard memlock unlimited
* soft memlock unlimited
EOF
# session courante
ulimit -l unlimited 2>/dev/null || true

# ── 4. validation NPU ────────────────────────────────────────────────────────
echo "[4/4] validation NPU…"
flm validate || {
  echo ""
  echo "Si 'Memlock Limit' != infinity : déconnecte/reconnecte la session (limits.d)."
  echo "Si le NPU n'est pas vu : vérifie 'amd_iommu' actif (pas amd_iommu=off)."
}

echo ""
echo "══════════════════════════════════════════════"
echo " FLM INSTALLÉ. Lance un modèle sur le NPU :"
echo "   flm run llama3.2:1b      # télécharge le kernel NPU depuis HuggingFace"
echo "   flm run qwen2.5:1.5b"
echo " (modèles du zoo NPU. Laneformer n'y est pas — arch custom.)"
echo "══════════════════════════════════════════════"
