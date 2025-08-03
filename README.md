# CRNN-LID – Language Identification basato su CRNN  

---

## 1 · Panoramica
Il progetto implementa un sistema di **Language Identification (LID)** che trasforma l’audio in spettrogrammi, li elabora con reti **Convolutional + Recurrent (CRNN)** e restituisce la lingua parlata.

**Lingue supportate (label predefinite)**  
| Codice | Lingua | Dataset di origine |
|--------|--------|--------------------|
| `EN`   | Inglese | VoxForge |
| `IT`   | Italiano | VoxForge |
| `SP`   | Spagnolo | VoxForge |
| `SR`   | Sardo (Sardinian) | VoxForge |

Puoi ampliare il numero di lingue aggiungendo altri CSV/ cartelle spettrogrammi e aggiornando `keras/config.yaml` (chiavi `label_names` e `num_classes`).

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
| **`keras/config.yaml`**                                         | Hyper-parametri, percorsi, mapping etichette | 🔑 file di configurazione centrale |
| **`requirements.txt`**                                          | Dipendenze Python (versioni da pin-nare) | consigliato virtualenv |
| **`LICENSE`**                                                   | MIT | |

---

## 3 · Script CLI & comandi di esempio

| Script                                                                                                                                    | Descrizione | Comando base |
|-------------------------------------------------------------------------------------------------------------------------------------------|-------------|--------------|
| **`manage/train.py`**                                                                                                                     | Addestra un modello da zero o finetune da pesi `.h5`  | `python manage/train.py --config manage/config.yaml [--weights start_weights.h5]` |
| **`manage/evaluate.py`**                                                                                                                   | Valuta un modello salvato, genera <br>‣ accuracy / precision / recall / F1<br>‣ matrice di confusione PNG<br>‣ curve ROC PNG<br>‣ report PDF | `python manage/evaluate.py --model trained_model.h5 --config manage/config.yaml [--testset]` |
| **`manage/predict.py`**                                                                                                                    | Predice lingua di **un singolo file audio** (qualsiasi formato gestito da SoX) | `python manage/predict.py --model trained_model.h5 --input speech.wav` |
| **`manage/tsne.py`**                                                                                                                       | Proietta features del penultimo layer in 2-D con t-SNE e salva scatter-plot | `python manage/tsne.py --model trained_model.h5 --config manage/config.yaml` |
| **`manage/visualize_conv.py`**                                                                                                             | Visualizza feature-maps di un layer convoluzionale | `python manage/visualize_conv.py --model trained_model.h5 --layer 3 --input sample.png` |
| **`data/wav_to_spectrogram.py`**                                                                                                          | Converte `.wav` → spettrogramma `.png` tramite SoX | `python data/wav_to_spectrogram.py --input wav_dir --output spec_dir` |
| **`data/create_csv.py`**                                                                                                                  | Genera CSV `<path,label>` da cartelle di spettrogrammi | `python data/create_csv.py --input spec_dir --output train.csv --label EN` |
| **`tools/convert_to_mono_wav.py`**                                                                                                        | Converte audio multi-canale→mono WAV (16 kHz) | `python tools/convert_to_mono_wav.py --input raw_dir --output mono_dir` |
| *(altri script in `tools/` come `clean_filenames.py`, `check_bad_images.py`, `audio_length.py`, ecc. sono self-documentati via `--help`)* | | |

> Tutti gli script hanno `--help` con dettagli parametri.

---

## 4 · Configurazione dettagliata (`keras/config.yaml`)

```yaml
train_data_dir: "train_data_dir/training.csv" # CSV <path,label>
validation_data_dir: "validation_data_dir/validation.csv"
test_data_dir: "test_data_dir/testing.csv"

batch_size: 
learning_rate: 
num_epochs:

data_loader: scegli uno dei loader in /keras/data_loaders
color_mode: "L"=grayscale, "RGB"=color
input_shape: [H, W, C]

model: scegli modello da /keras/models

segment_length: secondi per spettrogramma
pixel_per_second: risoluzione “orizzontale”

label_names: 
num_classes: