# Towards a robust out-of-the-box neural network model forgenomic data
Zhaoyi Zhang, Songyang Cheng, Claudia Solis-Lemus

## Citation

This repository contains the scripts for the [Zhang et al, 2021](https://arxiv.org/abs/2012.05995) manuscript:

```
@article{zhang2020towards,
  title={Towards a robust out-of-the-box neural network model for genomic data},
  author={Zhang, Zhaoyi and Cheng, Songyang and Solis-Lemus, Claudia},
  journal={arXiv preprint arXiv:2012.05995},
  year={2020}
}
```

## Data

We used publicly available data from the following manuscripts:

- Zeng H., Edwards M.D., Gifford D. K.(2015) "Convolutional Neural Network Architectures for Predicting DNA-Protein Binding".
Proceedings of Intelligent Systems for Molecular Biology (ISMB) 2016
Bioinformatics, 32(12):i121-i127. doi: 10.1093/bioinformatics/btw255.
[Motif data link](http://cnn.csail.mit.edu/),
[Paper link](https://pubmed.ncbi.nlm.nih.gov/27307608/)


- Nguyen, N.G., Tran, V.A., Ngo, D.L., Phan, D., Lumbanraja, F.R., Faisal, M.R., Abapihi, B., Kubo, M., Satou,
K. (2016) "DNA sequence classification by convolutional neural network". JBiSE 09(05), 280–286
[Splice data link](https://archive.ics.uci.edu/ml/datasets/Molecular+Biology+(Splice-junction+Gene+Sequences)),
[Histone data link](https://www.jaist.ac.jp/~tran/nucleosome/members.htm),
[Paper link](https://www.scirp.org/journal/paperinformation.aspx?paperid=65923)


## Scripts

### Pre-processing data

Python functions to download, clean and reformat the input data:
- [cnn/dna_nn/download.py](https://github.com/solislemuslab/dna-nn-theory/tree/master/cnn/dna_nn/download.py)
- [cnn/dna_nn/dataset.py](https://github.com/solislemuslab/dna-nn-theory/tree/master/cnn/dna_nn/dataset.py)

### CNN models

All scripts and output files corresponding to the CNN models are in the `cnn` folder.

- [cnn/dna_nn/model.py](https://github.com/solislemuslab/dna-nn-theory/tree/master/cnn/dna_nn/download.py) contains the CNN models
- Jupyter notebooks contain the reproducible steps to run the analyses on each of the datasets:
    - [cnn/histone.ipynb](https://github.com/solislemuslab/dna-nn-theory/blob/master/cnn/histone.ipynb)
    - [cnn/motif_discovery.ipynb](https://github.com/solislemuslab/dna-nn-theory/blob/master/cnn/motif_discovery.ipynb)
    - [cnn/splice.ipynb](https://github.com/solislemuslab/dna-nn-theory/blob/master/cnn/splice.ipynb)

### NLP models

All the scripts and output files corresponding to the NLP models are in the `nlp` folder. Jupyter notebooks contain the reproducible steps to run the analyses on each of the datasets.

#### LSTM-layer
- [nlp/uci_baseline_adam256.ipynb](https://github.com/solislemuslab/dna-nn-theory/tree/master/nlp/uci_baseline_adam256.ipynb)
- [nlp/histone_lstm_layer_adam.ipynb](https://github.com/solislemuslab/dna-nn-theory/tree/master/nlp/histone_lstm_layer_adam.ipynb)
- [nlp/histone_lstm_layer_sgd.ipynb](https://github.com/solislemuslab/dna-nn-theory/tree/master/nlp/histone_lstm_layer_sgd.ipynb)
- [nlp/discovery_baseline.py](https://github.com/solislemuslab/dna-nn-theory/tree/master/nlp/discovery_baseline.py)

#### doc2vec+NN
- [nlp/uci_doc2vec_save_embedding.ipynb](https://github.com/solislemuslab/dna-nn-theory/tree/master/nlp/uci_doc2vec_save_embedding.ipynb)
- [nlp/histone_doc2vec.ipynb](https://github.com/solislemuslab/dna-nn-theory/tree/master/nlp/histone_doc2vec.ipynb)
- [nlp/discovery_doc2vec.py](https://github.com/solislemuslab/dna-nn-theory/tree/master/nlp/discovery_doc2vec.py)

#### LSTM-AE+NN
- [nlp/uci_ae_adam32.ipynb](https://github.com/solislemuslab/dna-nn-theory/tree/master/nlp/uci_ae_adam32.ipynb)
- [nlp/uci_ae_adam256.ipynb](https://github.com/solislemuslab/dna-nn-theory/tree/master/nlp/uci_ae_adam256.ipynb)
- [nlp/histone_ae_adam32.ipynb](https://github.com/solislemuslab/dna-nn-theory/tree/master/nlp/histone_ae_adam32.ipynb)
- [nlp/histone_ae_adam256.ipynb](https://github.com/solislemuslab/dna-nn-theory/tree/master/nlp/histone_ae_adam256.ipynb)
- [nlp/histone_ae_adam1024.ipynb](https://github.com/solislemuslab/dna-nn-theory/tree/master/nlp/histone_ae_adam1024.ipynb)
- [nlp/discovery_ae.py](https://github.com/solislemuslab/dna-nn-theory/tree/master/nlp/discovery_ae.py)

### Figures

All figures in the manuscript were created with the R script in `plots/final-plots.Rmd`

## License

Our code is licensed under the
[MIT License](https://github.com/solislemuslab/dna-nn-theory/blob/master/LICENSE) &copy; Solis-Lemus lab projects (2021).


## Feedback, issues and questions

Please use the [GitHub issue tracker](https://github.com/solislemuslab/dna-nn-theory/issues) to report any issues or difficulties with the current code.