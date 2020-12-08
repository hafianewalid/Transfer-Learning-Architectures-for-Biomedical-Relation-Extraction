# Transfer-learning-architectures-for-biomedical-relations-extraction
## Running an experience
For running an experience run the main code on an CPU or GPU device with the according parameters.
### Display parameters list :

`$ python main.py -h`

### Add BioBert model :
To use BioBert model download the model folder and put it in the main folder : 

`$ gdown https://drive.google.com/uc?id=1OCpJn4k2986cwImRLMUCq2F6spUpt2t0`

`$ unzip biobert_v1.1_pubmed.zip` 

### Fine-tuning strategy examples :

* Fine-tuning on PgxCorpus

`$ python main.py -ft True -bert scibert -fine_tuning_model CNN_seg -F_type macro -corpus pgx -lr 3e-5 -num_ep 8`

* Fine-tuning on ChemProt

`$ python main.py -ft True -bert biobert -fine_tuning_model CNN_seg -F_type micro -corpus chemprot -lr 3e-5 -num_ep 8`

### Frozen strategy examples :

* Frozen on PgxCorpus

`$ python main.py -bert scibert -frozen_model CNN_RNN -F_type macro -corpus pgx -lr 0.001 -num_ep 30`

* Frozen on ChemProt

`$ python main.py -bert biobert -frozen_model CNN_RNN -F_type micro -corpus chemprot -lr 0.001 -num_ep 30`

## Experiences statistics 

### Results files 
The experiences results are saved into three types of files :
* .res : 
  - Contain the precision, recall and fscore (micro or macro).
  - You can use this file for statistical performances computing.
* .pred : 
  - Contain model predictions end ground truth. 
  - You can use this file for error analysing.
* .loss_acc: 
  - Contain the the accuracy and loss function value with the training data.
  - You can use this file for observe models convergences.

### Statistics computing 
To compute results statistics run the stat code.

`$ python stat_result.py`

The results should looks like the following example : 

```/////////////////////////// stat //////////////////////////
///////////////////////////////////////////////////////////
                min       max      mean       std
precision  0.729422  0.802180  0.781566  0.010286
recall     0.707363  0.818019  0.791717  0.013113
fscore     0.714244  0.804695  0.784400  0.010777
```

<img src="https://drive.google.com/uc?export=view&id=1BAApczvMWizF83cIrqtgbRBGNAa7bV6B" width="500" height="300">

## Data exploration 
In order to get an idea about data different methods of data exploration are implemented.

### Example of Data exploration :

`$ python data_analysis.py -corpus pgx`

<img src="https://drive.google.com/uc?export=view&id=11o50OrmDh1dx1CqfXj4sCVrxaYePjrwK" width="600" height="700">

## Embedding visualization 

The visualization of *BERT embedding vectors in each training epoch, might provide us an idea about how models learn.

### Example of embedding visualization :

`$python data_visualisation.py -bert scibert -corpus pgx`

<img src="https://drive.google.com/uc?export=view&id=1sMSGlvn89tonUs9OKgOzrZf2FFXQFV5K" width="600" height="700">




