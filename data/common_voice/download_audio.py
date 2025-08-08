import argparse
import inspect
import os
import random
import re
import shutil
import time
import traceback
from collections import deque
from typing import Optional, Dict, List, Deque, Tuple

import soundfile as sf
from datasets import load_dataset
from datasets.exceptions import DatasetNotFoundError
from tqdm import tqdm

ROOT_BASE = r"C:\Users\fraca\Documents\GitHub\crnn_lid\data\common_voice"

LANG_CODES: Dict[str, str] = {
    "english": "en",
    "spanish": "es",
    "italian": "it",
}

DEFAULT_MAX = 1500
DEFAULT_MIN_SECONDS = 5.0
DEFAULT_LOG_EVERY = 100

HF_DATASET_NS = "mozilla-foundation"
HF_DATASET_BASES: List[str] = [
    "common_voice_17_0",
    "common_voice_16_1",
    "common_voice_16_0",
    "common_voice_15_0",
    "common_voice_14_0",
]


def ensure_dirs(tmp_dir: str, out_dir: str) -> None:
    os.makedirs(tmp_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)


def _load_dataset_with_token(*args, token: Optional[str] = None, **kwargs):
    if token:
        sig = inspect.signature(load_dataset)
        if "token" in sig.parameters:
            kwargs["token"] = token
        elif "use_auth_token" in sig.parameters:
            kwargs["use_auth_token"] = token
    kwargs["trust_remote_code"] = True
    return load_dataset(*args, **kwargs)


def hf_pick_version(lang_code: str, forced_base: Optional[str], token: Optional[str], debug: bool) -> str:
    bases = [forced_base] if forced_base else HF_DATASET_BASES
    splits_try = ["validated", "test", "validation", "train"]
    errors: List[str] = []
    for base in bases:
        for split in splits_try:
            try:
                ds = _load_dataset_with_token(f"{HF_DATASET_NS}/{base}", lang_code, split=split, streaming=True,
                                              token=token)
                it = iter(ds)
                next(it)
                return base
            except (DatasetNotFoundError, ValueError, StopIteration) as e:
                if debug:
                    traceback.print_exc()
                errors.append(f"{base}/{split}: {e!r}")
                continue
    raise RuntimeError(
        "Nessuna versione Common Voice disponibile su Hugging Face per la lingua richiesta. Dettagli:\n" +
        "\n".join(errors)
    )


def duration_seconds(example: Dict) -> Optional[float]:
    val = example.get("duration")
    if val is not None:
        try:
            return float(val)
        except (TypeError, ValueError):
            pass
    audio = example.get("audio")
    if isinstance(audio, dict) and ("array" in audio) and ("sampling_rate" in audio):
        try:
            return float(len(audio["array"])) / float(audio["sampling_rate"])
        except (TypeError, ValueError):
            return None
    return None


def iter_hf_examples_streaming_random(lang_code: str, base: str, token: Optional[str],
                                      limit: int, min_seconds: float):
    splits = ["validated", "test", "validation", "train"]
    random.shuffle(splits)
    selected = 0
    with tqdm(total=limit, desc="Selezione casuale clip valide", unit="clip", dynamic_ncols=True) as pbar:
        for split in splits:
            ds = _load_dataset_with_token(f"{HF_DATASET_NS}/{base}", lang_code, split=split, streaming=True,
                                          token=token)
            for ex in ds:
                dur = duration_seconds(ex)
                if dur is None or dur < min_seconds:
                    continue
                sentence = ex.get("sentence") or ""
                yield ex, sentence
                selected += 1
                pbar.update(1)
                if selected >= limit:
                    return


def sanitize_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_")


def save_streaming_audio(example: Dict, tgt_path: str) -> Tuple[int, float]:
    audio = example.get("audio") or {}
    start = time.perf_counter()
    arr = audio.get("array")
    sr = audio.get("sampling_rate")
    if arr is None or sr is None:
        raise RuntimeError("Audio non disponibile in streaming")
    sf.write(tgt_path, arr, int(sr))
    elapsed = time.perf_counter() - start
    size = os.path.getsize(tgt_path)
    return size, elapsed


def mbps(bytes_count: int, seconds: float) -> float:
    if seconds <= 0:
        return 0.0
    return (bytes_count / (1024 * 1024)) / seconds


def process_hf_to_dir(
        lang_code: str,
        out_dir: str,
        prefix: str,
        min_seconds: float,
        max_clips: int,
        log_every: int,
        forced_base: Optional[str],
        no_token: bool,
        debug: bool,
) -> None:
    token = None if no_token else (os.getenv("HF_TOKEN") or None)
    print("Ricerca versione disponibile...")
    base = hf_pick_version(lang_code, forced_base, token, debug)
    print(f"Versione selezionata: {base}")

    manifest_rows: List[Dict[str, str]] = []
    window: Deque[Tuple[int, float]] = deque(maxlen=max(1, log_every))
    bytes_total = 0
    time_total = 0.0

    with tqdm(total=max_clips, desc="Salvataggio clip su disco", unit="clip", dynamic_ncols=True) as pbar_save:
        for ex, sentence in iter_hf_examples_streaming_random(lang_code, base, token, limit=max_clips,
                                                              min_seconds=min_seconds):
            dur = duration_seconds(ex)
            if dur is None or dur < min_seconds:
                continue

            orig_bn = os.path.basename(((ex.get("audio") or {}).get("path") or "").strip())
            base_name = os.path.splitext(orig_bn)[0]
            if not base_name:
                base_name = f"{int(time.time() * 1e6)}_{random.randint(0, 999999):06d}"
            base_name = sanitize_name(base_name)
            tgt_name = f"{prefix}-{base_name}.wav"
            tgt_path = os.path.join(out_dir, tgt_name)

            if not os.path.exists(tgt_path):
                sz, dt = save_streaming_audio(ex, tgt_path)
            else:
                sz = os.path.getsize(tgt_path)
                dt = 0.0

            manifest_rows.append({
                "path": tgt_name,
                "sentence": sentence or "",
                "duration": f"{dur:.3f}"
            })

            bytes_total += sz
            time_total += max(dt, 0.0)
            window.append((sz, dt))
            pbar_save.update(1)

            if log_every > 0 and len(manifest_rows) % log_every == 0:
                w_bytes = sum(b for b, _ in window)
                w_time = sum(t for _, t in window)
                avg_window = mbps(w_bytes, w_time)
                avg_global = mbps(bytes_total, time_total)
                pbar_save.write(
                    f"[{len(manifest_rows)} clip] Velocità media finestra: {avg_window:.2f} MB/s | "
                    f"Media cumulata: {avg_global:.2f} MB/s"
                )

            if len(manifest_rows) >= max_clips:
                break

    if manifest_rows:
        tsv_path = os.path.join(out_dir, f"{prefix}-manifest.tsv")
        with open(tsv_path, "w", encoding="utf-8", newline="") as f:
            cols = ["path", "sentence", "duration"]
            f.write("\t".join(cols) + "\n")
            for r in manifest_rows:
                f.write("\t".join(r.get(c, "") for c in cols) + "\n")

    print(f"  Estratte {len(manifest_rows)} clip >= {min_seconds:.1f}s.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Scarica ed estrae clip da Common Voice.")
    parser.add_argument("language", choices=LANG_CODES.keys())
    parser.add_argument("--max", type=int, default=DEFAULT_MAX, metavar="N", help="Numero massimo di clip da scaricare")
    parser.add_argument("--min-seconds", type=float, default=DEFAULT_MIN_SECONDS)
    parser.add_argument("--log-every", type=int, default=DEFAULT_LOG_EVERY, help="Log velocità media ogni N clip.")
    parser.add_argument("--base", choices=HF_DATASET_BASES, default=None)
    parser.add_argument("--no-token", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    lang_code = LANG_CODES[args.language]
    tmp_dir = os.path.join(ROOT_BASE, "tmp")
    out_dir = os.path.join(ROOT_BASE, args.language)
    ensure_dirs(tmp_dir, out_dir)

    prefix = f"{lang_code}"
    print(f"Estrazione e filtro clip (durata minima {args.min_seconds}s, max {args.max} clip)...")
    try:
        process_hf_to_dir(
            lang_code=lang_code,
            out_dir=out_dir,
            prefix=prefix,
            min_seconds=args.min_seconds,
            max_clips=args.max,
            log_every=max(1, args.log_every),
            forced_base=args.base,
            no_token=args.no_token,
            debug=args.debug,
        )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    print("Fatto – audio salvato in", out_dir)


if __name__ == "__main__":
    main()
