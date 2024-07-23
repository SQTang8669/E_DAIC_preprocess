# E-DAIC dataset preprocess

This is a simple project to clean and filter the original files in [E-DAIC dataset](https://dcapswoz.ict.usc.edu/), the original data struture should be:

```
-301_P
    -301_Transcript.csv
    -310_AUDIO.wav
```

# steps:
- step 1: change the data structure easier to view and process like this
```
-trans
    -301.json
    -302.json
    -...

-audio
    -301.wav
    -302.wav
    -...
```

- step 2: filter out the interviewer's voice using [DeepFilterNet](https://github.com/Rikorose/DeepFilterNet).

OR simply use Whisper to transcrible the audio files -,-

- step 3.1: segment audio files using original json files, automatically correct errors like starting time is bigger than ending time, ending time is bigger than the audio length.

- step 3.2: feed each segment to [whisper](https://github.com/openai/whisper) and replace the original text with new ones.

# usage
