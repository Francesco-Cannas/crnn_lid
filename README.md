# CRNN-LID – Language Identification basato su CRNN  

---

## 1 · Panoramica
Il progetto implementa un sistema di **Language Identification (LID)** che trasforma l’audio in spettrogrammi, li elabora con reti **Convolutional + Recurrent (CRNN)** e restituisce la lingua parlata.

**Lingue supportate (label predefinite)**  
| Codice | Lingua | Dataset di origine |
|--------|--------|--------------------|
| `EN`   | Inglese | CommonVoice |
| `IT`   | Italiano | CommonVoice |
| `SP`   | Spagnolo | CommonVoice |
| `SR`   | Sardo | CommonVoice |

Puoi ampliare il numero di lingue aggiungendo altri CSV/ cartelle spettrogrammi e aggiornando `manage/config.yaml` (chiavi `label_names` e `num_classes`).

---

## 2 · Struttura del repository

| Cartella / file                                                 | Contenuto principale | Note operative |
|-----------------------------------------------------------------|----------------------|----------------|
| **`/manage/`**                                                  | Tutto il codice di training, inferenza e valutazione | vedi § 3 |
| **`/manage/models/`**                                           | Architetture: `crnn.py`, `topcoder_crnn*.py`, `inceptionv3_crnn.py`, `resnet.py`, … | seleziona il modello tramite `config.yaml → model` |
| **`/manage/data_loaders/`**                                     | Loader & generatori: `ImageLoader`, `DirectoryLoader`, `SpectrogramGenerator`, … | si specifica con `config.yaml → data_loader` |
| **`/data/`**                                                    | Script per scaricare corpora e generare spettrogrammi (`wav_to_spectrogram.py`, `create_csv.py`) | serve SoX installato |
| **`/tools/`**                                                   | Utility varie su file audio (pulizia nomi, normalizzazione, check immagini, …) | script CLI autonomi |
| **`train_data_dir/`, `validation_data_dir/`, `test_data_dir/`** | CSV con `<path_spettrogramma,label>` | usati dai loader |
| **`manage/config.yaml`**                                        | Hyper-parametri, percorsi, mapping etichette | 🔑 file di configurazione centrale |
| **`requirements.txt`**                                          | Dipendenze Python (versioni da pin-nare) | consigliato virtualenv |
| **`LICENSE`**                                                   | MIT | |

---

## 3 · Script CLI & comandi di esempio

| Script                                                                                                                                           | Descrizione | Comando base                                                                                         |
|--------------------------------------------------------------------------------------------------------------------------------------------------|-------------|------------------------------------------------------------------------------------------------------|
| **`manage/train.py`**                                                                                                                            | Addestra un modello da zero o finetune da pesi `.h5`  | `python -m crnn_lid.manage.train --config config.yaml`                                               |
| **`manage/evaluate.py`**                                                                                                                         | Valuta un modello salvato, genera <br>‣ accuracy / precision / recall / F1<br>‣ matrice di confusione PNG<br>‣ curve ROC PNG<br>‣ report PDF | `python -m crnn_lid.manage.evaluate --config config.yaml [--testset]`                                |
| **`manage/predict.py`**                                                                                                                          | Predice lingua di **un singolo file audio** (qualsiasi formato gestito da SoX) | `python -m crnn_lid.manage.predict --model trained_model.h5 --input speech.wav --config config.yaml` |
| **`manage/tsne.py`**                                                                                                                             | Proietta features del penultimo layer in 2-D con t-SNE e salva scatter-plot | `python -m crnn_lid.manage.tsne --config config.yaml`                                                |
| **`manage/visualize_conv.py`**                                                                                                                   | Visualizza feature-maps di un layer convoluzionale | `python -m crnn_lid.manage.visualize_conv --config config.yaml`                                      |
| **`data/wav_to_spectrogram.py`**                                                                                                                 | Converte `.wav` → spettrogramma `.png` tramite SoX | `python -m crnn_lid.data.wav_to_spectrogram`                                                         |
| **`tools/convert_to_mono_wav.py`**                                                                                                               | Converte audio multi-canale→mono WAV (16 kHz) | `python -m crnn_lid.tools.convert_to_mono_wav --path ...`                                            |
| *(altri script in `tools/` come `clean_filenames.py`, `check_bad_images.py`, `audio_length.py` e `neg23.py` sono self-documentati via `--help`)* | |                                                                                                      |

> Tutti gli script hanno `--help` con dettagli parametri.

---

## 4 · Configurazione dettagliata (`manage/config.yaml`)

```yaml
train_data_dir: "train_data_dir/training.csv" 
validation_data_dir: "validation_data_dir/validation.csv"
test_data_dir: "test_data_dir/testing.csv"

batch_size: 
learning_rate: 
num_epochs:

data_loader: scegli uno dei loader in /manage/data_loaders
color_mode: "L"=grayscale, "RGB"=color
input_shape: [H, W, C]

model: scegli modello da /manage/models

segment_length: secondi per spettrogramma
pixel_per_second: risoluzione “orizzontale”

label_names: 
num_classes: