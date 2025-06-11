#!/bin/bash

# ==== Настройки ====
SRC_DIR="/mnt/f/_prg/python/ComfyUI_Gayrat"
CONTAINER_NAME="comfyui-union2"
DEST_PATH="/workspace/ComfyUI/custom_nodes/ComfyUI_Gayrat"

# ==== Временные пути ====
TMP_DIR="/mnt/d/_tmp/ComfyUI_Gayrat_sync"

# ==== Исключаемые файлы и папки ====
EXCLUDE_OPTS=(
  --exclude='.git'
  --exclude='.github'
  --exclude='.idea'
  --exclude='.venv'
)

# ==== Очистка старых временных данных ====
if [ -d "$TMP_DIR" ]; then
  rm -rf "$TMP_DIR"
fi

mkdir -p "$TMP_DIR"

# ==== Копируем с исключениями ====
echo "🔍 Копирую файлы локально с исключениями..."
rsync -av "${EXCLUDE_OPTS[@]}" "$SRC_DIR/" "$TMP_DIR/"

# ==== Копируем в контейнер ====
echo "🐋 Копирую файлы в контейнер $CONTAINER_NAME..."
docker exec comfyui-union2 rm -rf /workspace/ComfyUI/custom_nodes/ComfyUI_Gayrat/
docker exec -it "$CONTAINER_NAME" mkdir -p "$DEST_PATH"
docker cp "$TMP_DIR/." "$CONTAINER_NAME:$DEST_PATH"

# ==== Очистка ====
echo "🧹 Очистка временных файлов..."
rm -rf "$TMP_DIR"

echo "✅ Готово! Файлы успешно скопированы в контейнер:"
echo "   $CONTAINER_NAME:$DEST_PATH"