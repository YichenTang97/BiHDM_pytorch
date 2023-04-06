# BiHDM_pytorch
 An unofficial pytorch implementation of the BiHDM model proposed by Yang et al. [1] for decoding emotion from multi-channel electroencephalogram (EEG) recordings, with scikit-learn compatibility.

> **Warning** 
> :exclamation: The Domain Adversarial Strategy as described in section II.C of [1] is not implemented - this will be added to the current implementation soon.

> **Warning**
> Please note this is not an official implementation, nor has been tested on the datasets used in the original studies. Due to different libraries and hyperparameters used in the implementation (and potentially implementation errors), there might be differences in the performance of this model to the ones as described in the papers. Please always examine the source code, make your own changes if necessary, and describe the actual implementation if you are using this model for an academic study. And please raise an issue if you found any implementation error in my code, thank you!

## Introduction

This repository presents a pytorch implementation of the BiHDM model proposed by Yang et al. [1]. The BiHDM model effectively leverages the bi-hemispheric discrepancy features of EEG to achieve high classification accuracies in decoding emotions. 

The BiHDM model first obtains deep representations for electrodes on the left and right hemispheres separately, utilizing either a horizontal or a vertical stream. Within each stream, the model learns bi-hemisphere discrepancy features by performing pairwise operations on the deep representations for matching electrodes on the two hemispheres. Finally, the bi-hemisphere discrepancy features from both horizontal and vertical streams are combined to predict the emotion label for the given EEG sample (see Fig. 1 in [1]).

The default hyper-parameters utilized in this implementation are based on the settings outlined in the original paper [1]. While these settings performed generally well with my own EEG datasets, tuning certain hyperparameters did lead to improved classification accuracies. When applying this implementation for your own projects, you may want to experiment with these settings for best outcomes.

## Requirements
This model was coded and tested on Python 3.9 with the following libraries and versions (minor differences in versions should not affect the model outcomes):

```Python
numpy >= 1.21.6
scikit-learn >= 1.1.3
torch == 1.13.1+cu116
```

## Examples

See "BiHDM_example.ipynb".

```Python
>>> import numpy as np
>>> from sklearn.datasets import load_digits
>>> from sklearn.model_selection import cross_val_score

>>> from BiHDM import BiHDMClassifier

>>> # Define 64 EEG channels using 10-20 standard
>>> ch_names = ['Fp1', 'AF7', 'AF3', 'F1', 'F3', 'F5', 'F7', 'FT7', 'FC5', 'FC3', 'FC1', 
>>>             'C1', 'C3', 'C5', 'T7', 'TP7', 'CP5', 'CP3', 'CP1', 'P1', 'P3', 'P5', 
>>>             'P7', 'P9', 'PO7', 'PO3', 'O1', 'Iz', 'Oz', 'POz', 'Pz', 'CPz', 'Fpz', 
>>>             'Fp2', 'AF8', 'AF4', 'AFz', 'Fz', 'F2', 'F4', 'F6', 'F8', 'FT8', 'FC6', 
>>>             'FC4', 'FC2', 'FCz', 'Cz', 'C2', 'C4', 'C6', 'T8', 'TP8', 'CP6', 'CP4', 
>>>             'CP2', 'P2', 'P4', 'P6', 'P8', 'P10', 'PO8', 'PO4', 'O2']

>>> lh_chs = ['Fp1', 'AF7', 'AF3', 'F7', 'F5', 'F3', 'F1', 'FT7', 'FC5', 'FC3', 'FC1', 
>>>           'T7', 'C5', 'C3', 'C1', 'TP7', 'CP5', 'CP3', 'CP1', 'P7', 'P5', 'P3', 'P1', 
>>>           'PO7', 'PO3', 'O1']
>>> rh_chs = ['Fp2', 'AF8', 'AF4', 'F8', 'F6', 'F4', 'F2', 'FT8', 'FC6', 'FC4', 'FC2', 
>>>           'T8', 'C6', 'C4', 'C2','TP8', 'CP6', 'CP4', 'CP2', 'P8', 'P6', 'P4', 'P2', 
>>>           'PO8', 'PO4', 'O2']
>>> lv_chs = ['Fp1', 'AF7', 'F7', 'FT7', 'T7', 'TP7', 'P7', 'PO7', 'AF3', 'F5', 'FC5', 
>>>           'C5', 'CP5', 'P5', 'O1', 'F3', 'FC3', 'C3', 'CP3', 'P3', 'PO3', 'F1', 'FC1', 
>>>           'C1', 'CP1', 'P1']
>>> rv_chs = ['Fp2', 'AF8', 'F8', 'FT8', 'T8', 'TP8', 'P8', 'PO8', 'AF4', 'F6', 'FC6', 
>>>           'C6', 'CP6', 'P6', 'O2', 'F4', 'FC4', 'C4', 'CP4', 'P4', 'PO4', 'F2', 'FC2', 
>>>           'C2', 'CP2', 'P2']

>>> # Generate some data for classification
>>> X = np.ones((1000, 64, 5)) # 1000 samples x 64 channels x 5 bands per channel (delta, theta, alpha, beta, gamma)
>>> y = np.repeat([0,1], 500)

>>> # Let's simulate a frontal alpha-asymmetry for the classifier to learn from
>>> left_frontal_chs = ['Fp1', 'AF3', 'AF7', 'F1', 'F3', 'F5', 'FC3', 'FC1']
>>> X[:500,np.isin(ch_names, left_frontal_chs),2] -= 1

>>> # And let's add some gaussian noise
>>> rng = np.random.default_rng(42)
>>> X += rng.normal(scale=0.5, size=X.shape)

>>> # Reshape X to meet sklearn standard
>>> X = X.reshape(1000, -1)


>>> clf = BiHDMClassifier(ch_names, lh_chs, rh_chs, lv_chs, rv_chs, 
>>>                     d_stream=32, d_pair=32, d_global=32, d_out=16, 
>>>                     k=6, a=0.01, pairwise_operation='subtraction', 
>>>                     rnn_stream_kwargs={}, rnn_global_kwargs={}, 
>>>                     loss='NLLLoss', optimizer='SGD', lr=0.003,
>>>                     epochs=8, batch_size=200, loss_kwargs={}, 
>>>                     optimizer_kwargs=dict(momentum=0.9, weight_decay=0.95),
>>>                     random_state=42, use_gpu=True, verbose=False)

>>> pipeline = Pipeline([
>>>     ('scaler', StandardScaler()),
>>>     ('BiHDM', clf)
>>> ])

>>> scores = cross_val_score(clf, X, y)
>>> print(np.mean(scores))
0.783
>>> print(scores)
[0.745 0.76  0.775 0.805 0.83 ]
```

# Acknowledgements
 Special thanks to some partial implementation of BiHDM by https://github.com/numediart, which inspired some of my implementation.

# References
 [1] Y. Li et al., “A Novel Bi-Hemispheric Discrepancy Model for EEG Emotion Recognition,” IEEE Trans. Cogn. Dev. Syst., vol. 13, no. 2, pp. 354–367, Jun. 2021, doi: 10.1109/TCDS.2020.2999337.
