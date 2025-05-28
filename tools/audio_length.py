import os
import subprocess
import sys
import shlex

def get_audio_length(f):
  command = f"soxi -D {shlex.quote(f)}"
  return float(subprocess.check_output(command, shell=True))

if __name__ == '__main__':
  target_dir = sys.argv[1]
  files = [f for f in os.listdir(target_dir) if os.path.splitext(f)[1] == ".wav"]
  total = sum(get_audio_length(os.path.join(target_dir, f)) for f in files)
  print(total / 60. / 60.)