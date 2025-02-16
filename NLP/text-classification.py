from datasets import load_dataset
import pandas as pd
import matplotlib.pyplot as plt

dataset_url = "https://huggingface.co/datasets/transformersbook/emotion-train-split/raw/main/train.txt"
emotions = load_dataset("csv", data_files=dataset_url, sep=";", names=["text", "label"])

emotions.set_format(type="pandas")
df = emotions["train"][:]

df["label"].value_counts(ascending=True).plot.barh()

df["Words Per Tweet"] = df["text"].str.split().apply(len)
df.boxplot("Words Per Tweet", by="label", grid=False, showfliers=False, color="black")
plt.suptitle("")
plt.xlabel("")
plt.show()