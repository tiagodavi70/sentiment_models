# Sentiment Models


Fast iteration for training of sentiment analysis Models

# Build images with network

``` bash
# add network
```

# Preparing dataset

```bash
# -f - the dataset file path from kaggle
# -o directory output
python convert_from_csv.py -f fer2013.csv -o dataset
```

# Run Image Training Docker Container

* Jupyter Notebook
``` bash
sudo docker run --runtime=nvidia -it --net=host -e NVIDIA_VISIBLE_DEVICES=0 -v ~/Documents/sentiment_models:/home/ tf2.0 jupyter notebook
```

* Bash
``` bash
sudo docker run --runtime=nvidia -it --net=host -e NVIDIA_VISIBLE_DEVICES=0 -v ~/Documents/sentiment_models:/home/ tf2.0 /bin/bash
```

# Run Multimodal Training Docker Container


* Bash
``` bash
sudo docker run --runtime=nvidia -it --net=host -e NVIDIA_VISIBLE_DEVICES=0 -v ~/Documents/sentiment_models/multimodal:/home/ --user="root" multimodal_sentiment /bin/bash
```


# To Do

Revise on dockerfiles:
ERROR: google-auth 1.10.0 has requirement setuptools>=40.3.0, but you'll have setuptools 36.4.0 which is incompatible.
ERROR: tensorboard 2.0.2 has requirement setuptools>=41.0.0, but you'll have setuptools 36.4.0 which is incompatible.

variable input size for audio (RNN)
audio duration on preprocess audio script
