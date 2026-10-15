#!/usr/bin/env bash
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader
echo "cached models:"; ls ~/.cache/huggingface/hub | grep -E "^models--" | sed 's/models--//; s/--/\//'
echo "disk:"; df -h ~ | tail -1
echo "weights present for codellama-7b?"; ls ~/.cache/huggingface/hub/models--codellama--CodeLlama-7b-hf/snapshots/*/ 2>/dev/null | head -8
