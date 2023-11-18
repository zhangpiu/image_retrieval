# Image Retrieval
This is a project for image retrieval implemented in C++/Python. It consists of two main parts:
feature extraction and vector retrieval.

# Prerequisites
## C++
``` bash

```

## Python
``` bash
```

# Getting Started
## ImageNet-1K
Download ImageNet-1K from [Hugging Face](https://huggingface.co/datasets/imagenet-1k)

## Feature Extraction
Extract image features using pre-trained ResNet50.
``` bash
python image_retrieval/feature_extraction/inferencer.py \
    -d /path/to/imagenet_1k_rawimgs \
    -i /path/to/imagenet_1k_rawimgs/train.txt \
    -o data.pb \
    -j image_retrieval/feature_extraction/resnet50.py \
    -w 8 \
    -b 64
```

## Vector Search
Once the features are extracted, they are transformed into a vector representation.
This allows for efficient indexing and searching of images based on their visual similarities.

```bash
./image_retrieval/ann/search_engine -i data.pb -p 8001
```

## Demo UI
``` bash
python image_retrieval/demo.py 8000 -t localhost:8001 --resource /path/to/imagenet_1k_rawimgs
```
