# Interpretable Textual Neuron Representations for NLP

Implementation from [Poerner, N., Roth, B., Schütze, H. (2018). Interpretable Textual Neuron Representations for NLP. EMNLP.](https://aclanthology.org/W18-5437/), original code can be found [here](https://github.com/NPoe/input-optimization-nlp).

## Data
We use the dataset from [Chrupała, G., Gelderloos, L., & Alishahi, A. (2017). Representations of language in a model of visually grounded speech signal. ACL.](http://www.aclweb.org/anthology/P17-1057), which can be downloaded [here](https://zenodo.org/record/804392/files/data.tgz).

## Code

`train.py`

Trains a keras implementation of the GRU IMAGINET architecture from [Kádár, A., Chrupała, G., & Alishahi, A. (2017). Representation of linguistic form and function in recurrent neural networks. Computational Linguistics.](http://www.aclweb.org/anthology/J17-4003). 

`optimize.py [search|embfree|logits|softmax|gumbel] [I|T]`

Finds optimal n-gram for the image or text modality using the specified method.
