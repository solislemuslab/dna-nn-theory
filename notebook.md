# Data

- DNA-protein binding [paper](http://cnn.csail.mit.edu/): sequence length 101, ~2000 samples
- 12 datasets in Nguyen2016 [paper](https://www.scirp.org/pdf/JBiSE_2016042713533805.pdf): sequence length 500 for most datasets, and sample size varying in the thousands
- COVID genomes [here](https://www.cogconsortium.uk/data/) ~14000 aligned sequences


# Method alternatives

- Zeng2016
    - 4xL matrix (L=101) converted to one-hot encoding
    - Convolutional NN (window size=24)
    - Slow to train, only 50% accuracy with 1 epoch

- Nguyen2016
    - kmer table (k=3), word2vec
    - Convolutional NN (window size=2)
    - trains in minutes and has high accuracy (train 97%, test 80%)