# This folder contains several scripts for getting and preparing training data

## Requirements

For downloading data you will need to install `Firefox` and further tools with `pip install -r requirements.txt`

### Voxforge
- Downloads the audio samples from www.voxforge.org for some languages
```bash
voxforge/download-data.sh
voxforge/extract_tgz.sh {path_to_german.tgz} german
```

## Convert Audio Files to Spectrograms

After you've downloaded the audio files you can convert them to spectogram images.
Make sure you have [SoX](http://sox.sourceforge.net/) installed. To create 500x129x1 grayscale spectrogram images run the following script.

```
python /data/wav_to_spectrogram.py --source <path> --target <path>
```

The above script uses different spectrogram generators to augment the data with additional noise or background music if needed. Adjust the imports accordingly. Also make sure to adjust the languages if you are not using all languages indicated in the code.