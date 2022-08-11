#!/usr/bin/env bash
set -xe
WHEEL_FILE=$1
DEST_DIR=$2

CUDA_SUFFIX=cu$(echo "$CUDA_VERSION" | tr '.' '_')
WHEEL_FILENAME=$(basename "${WHEEL_FILE}")
DEST_FILENAME=$(echo "${WHEEL_FILENAME}" | sed "s#torchrec_dynamic_embedding#torchrec_dynamic_embedding_${CUDA_SUFFIX}#g")
mv  "${WHEEL_FILE}" "${DEST_DIR}/${DEST_FILENAME}"
