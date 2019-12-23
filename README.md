# Sentiment Models


Fast iteration for training of sentiment analysis Models


# Preparing dataset

```bash
# -f - the dataset file path from kaggle
# -o directory output
python convert_from_csv.py -f fer2013.csv -o dataset
```

# Run Training Docker Container

* Jupyter Notebook
``` bash
sudo docker run --runtime=nvidia -it --net=host -e NVIDIA_VISIBLE_DEVICES=0 -v ~/Documents/sentiment_models:/home/ tf2.0 jupyter notebook
```

* Bash
``` bash
sudo docker run --runtime=nvidia -it --net=host -e NVIDIA_VISIBLE_DEVICES=0 -v ~/Documents/sentiment_models:/home/ tf2.0 /bin/bash
```
