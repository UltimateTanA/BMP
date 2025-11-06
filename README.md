Steps to try enhanced BERT tailored towards hatespeech classification:

1. Install Python 2.7.18
2. Run prerequisites.py (python3 prerequisites.py)
   This generates requirements.txt (required libraries to run the final training loop) and USAGE_GUIDE.txt with detailed information on our implementation.
3. Follow the output from the previous step and install the required libraries from the requirements.txt file.
4. Ensure that you have a dataset ready with 2 columns: text and label. Ensure that it has the name hateSpeechDataset.csv
5. Run hate_speech_classifier.py (python3 hate_speech_classifier.py). This will initiate the pipeline for dataset splitting, BERT training, and final testing.
