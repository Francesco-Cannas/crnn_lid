if [ $# -lt 2 ]; then
  echo "Usage: $0 <tgz file> <language>"
  exit 1
fi

ZIP=$1
LANG=$2
TEMP_DIR=$(dirname "$ZIP")
FILE_NAME=$(basename "$ZIP")
STRIPPED_FILE_NAME=${FILE_NAME%.tgz}

tar -xf "$ZIP" -C "$TEMP_DIR"

UNZIPPED_FOLDER="$TEMP_DIR/$STRIPPED_FILE_NAME"
WAVES="$UNZIPPED_FOLDER/wav"

mkdir -p "$LANG"

for WAVE in "$WAVES"/*; do
  if [ -f "$WAVE" ]; then
    duration=$(ffprobe -i "$WAVE" -show_entries format=duration -v quiet -of csv="p=0")
    duration_int=${duration%.*}
    if [ "$duration_int" -gt 2 ]; then
      mv "$WAVE" "$LANG/$STRIPPED_FILE_NAME-$(basename "$WAVE")"
    else
      echo "Skipping $WAVE (duration ${duration}s)"
      rm "$WAVE"
    fi
  fi
done

rm -f "$ZIP"
rm -rf "$UNZIPPED_FOLDER"