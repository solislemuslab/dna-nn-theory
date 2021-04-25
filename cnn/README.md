The result directory contains all the result files and the combined accuracy file. The result files have the following format `<model>-<dataset>-<metric>.csv`.

`<model>: {cnn_deepdbp, cnn_nguyen_2_conv2d, cnn_nguyen_conv1d_2_conv2d, cnn_zeng_2_conv2d, cnn_zeng_3_conv2d, cnn_zeng_4_conv2d, cnn_zeng_4_conv2d_l2, deepram_conv1d_embed, deepram_conv1d_onehot, deepram_conv1d_recurrent_embed, deepram_conv1d_recurrent_onehot, deepram_recurrent_embed, deepram_recurrent_onehot}`

`<dataset>: {histone, motif_discovery, splice}`

`<metric>: {accuracy, dynamics, pr, roc}`