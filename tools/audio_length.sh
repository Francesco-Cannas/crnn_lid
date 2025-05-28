#!/bin/bash

if [ -z "$1" ]; then
  echo "Devi specificare la cartella da analizzare"
  exit 1
fi

TmpFile=tmp.txt

find "$1" -name "*.wav" -exec soxi -D {} \; > $TmpFile
awk '{ sum += $1 } END { print sum }' $TmpFile
rm $TmpFile