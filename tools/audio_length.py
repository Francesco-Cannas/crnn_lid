import subprocess
import sys
from pathlib import Path


def get_audio_length(wav_path: Path) -> float:
  result = subprocess.check_output(["soxi", "-D", str(wav_path)], text=True)
  return float(result.strip())


def hms(seconds: float) -> tuple[int, int, int]:
  h = int(seconds // 3600)
  m = int((seconds % 3600) // 60)
  s = int(seconds % 60)
  return h, m, s


if __name__ == "__main__":
  target_dir = Path(sys.argv[1])

  wav_files = list(target_dir.rglob("*.wav"))
  total_sec = sum(get_audio_length(p) for p in wav_files)
  avg_sec = total_sec / len(wav_files) if wav_files else 0

  th, tm, ts = hms(total_sec)
  ah, am, _ = hms(avg_sec)

  print("📂 Cartella analizzata :", target_dir)
  print("🎧 File WAV trovati    :", len(wav_files))
  print(f"⏲️  Durata totale      : {th:02d}:{tm:02d}:{ts:02d} (hh:mm:ss) ≈ {total_sec / 3600:.2f} ore")
  print(f"📏 Durata media/file   : {ah:02d}:{am:02d} (hh:mm) ≈ {avg_sec / 60:.1f} minuti")
