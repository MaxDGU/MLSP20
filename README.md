In this project I feed several language networks (LSTM's, RNN's) with jazz chord data to evaluate their effectiveness at predicting the next group of chords (2,4, and 8 beats) in a song. Research project on AI-assisted music making with the Institute for Audio Acoustic Music Research (IRCAM) in Paris. 

See Utilities --> SaveSetGroupBy.py for my individual contributions to the data ingestion process, expanding chord memory from 2 beats to 4,6, and 8 beats into a given jazz song. 

This feature represents the main data ingestion upgrade from MLSP19.  

Create dataset -> buildDataset.py  
Train model -> train.py  
Test set of models -> combineTestFile.py  
