# attention_is_all_you_need
A repository containing the implementation of [Attention is All You Need](https://arxiv.org/abs/1706.03762) paper. Text classification, text generation, and other tasks mentioned in the paper

## Setup

```
git clone https://github.com/sovit-123/attention_is_all_you_need.git
pip install requirements.txt
pip install .
```

## Executing Examples

* All example notebooks/scripts are in the `examples` directory.
* All models are in the `attention` directory.
* All examples are self contained inside their respective directories in the `examples` folder except a few imports from the common `examples/utils` directory.

## Data

* Datasets smaller than 25 MB are in the `examples/data` directory.
* Larger datasets are either automatically downloaded or need to manually put in the root folder `input` directory.
