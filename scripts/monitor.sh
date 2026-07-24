#!/usr/bin/env bash
set -euo pipefail
watch -n 2 'nvidia-smi; echo; ps -eo pid,etimes,%cpu,%mem,cmd --sort=-%cpu | head -25'
