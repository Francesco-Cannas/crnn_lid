import mimetypes
import os
import subprocess
import sys


def r128Stats(filePath):
    ffargs = [
        'ffmpeg', '-nostats', '-i', filePath,
        '-filter_complex', 'ebur128', '-f', 'null', '-'
    ]
    proc = subprocess.Popen(ffargs, stderr=subprocess.PIPE)
    stats = proc.communicate()[1].decode('utf-8', errors='replace')

    summaryIndex = stats.rfind('Summary:')
    if summaryIndex == -1:
        raise ValueError("Non è stato trovato il blocco Summary nel log ffmpeg.")

    summaryList = stats[summaryIndex:].split()

    return {
        'I': float(summaryList[summaryList.index('I:') + 1]),
        'I Threshold': float(summaryList[summaryList.index('I:') + 4]),
        'LRA': float(summaryList[summaryList.index('LRA:') + 1]),
        'LRA Threshold': float(summaryList[summaryList.index('LRA:') + 4]),
        'LRA Low': float(summaryList[summaryList.index('low:') + 1]),
        'LRA High': float(summaryList[summaryList.index('high:') + 1])
    }

def linearGain(iLUFS, goalLUFS=-23):
    gainLog = -(iLUFS - goalLUFS)
    return 10 ** (gainLog / 20)


def ffApplyGain(inPath, outPath, linearAmount):
    ffargs = ['ffmpeg', '-y', '-i', inPath, '-af', f'volume={linearAmount}']
    if outPath.lower().endswith('.mp3'):
        ffargs += ['-acodec', 'libmp3lame', '-aq', '0']
    ffargs.append(outPath)
    subprocess.run(ffargs, stderr=subprocess.PIPE)


def notAudio(filePath):
    if os.path.basename(filePath).startswith("audio"):
        return True
    thisMime = mimetypes.guess_type(filePath)[0]
    return thisMime is None or not thisMime.startswith("audio")


def neg23Directory(directoryPath):
    for thisFile in os.listdir(directoryPath):
        thisPath = os.path.join(directoryPath, thisFile)
        if notAudio(thisPath):
            continue
        neg23File(thisPath)
    print("Batch complete.")

def neg23File(filePath):
    if notAudio(filePath):
        print(f"'{filePath}' non è un file audio valido.")
        return False

    print(f"Scanning {filePath} for loudness...")

    try:
        loudnessStats = r128Stats(filePath)
    except Exception as e:
        print(f"Errore durante la scansione: {e}")
        return False

    gainAmount = linearGain(loudnessStats['I'])

    outputDir = os.path.join(os.path.dirname(filePath), "neg23")
    os.makedirs(outputDir, exist_ok=True)

    outputPath = os.path.join(outputDir, os.path.basename(filePath))
    print(f"Creazione del file normalizzato a -23 LUFS in: {outputPath}")

    try:
        ffApplyGain(filePath, outputPath, gainAmount)
    except Exception as e:
        print(f"Errore durante l'applicazione del gain: {e}")
        return False

    print("Fatto.")
    return True


if __name__ == "__main__":
    if len(sys.argv) == 2 and os.path.isdir(sys.argv[1]):
        neg23Directory(sys.argv[1])
    elif len(sys.argv) == 1:
        neg23Directory(os.getcwd())
    elif len(sys.argv) == 2 and os.path.isfile(sys.argv[1]):
        neg23File(sys.argv[1])
    elif len(sys.argv) == 2 and os.path.isfile(os.path.join(os.getcwd(), sys.argv[1])):
        neg23File(os.path.join(os.getcwd(), sys.argv[1]))
    else:
        correctUsage = (
            "Uso corretto:\n"
            "  neg23 somefile.wav\n"
            "  neg23 /directory/da/processare/"
        )
        print(correctUsage)