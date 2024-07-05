from pymongo import MongoClient, collection
import numpy as np
from matplotlib import pyplot as plt
from typing import List, Iterable
from numpy.typing import ArrayLike


def connect():
    client = MongoClient("mongodb://root:secure_pw@localhost:27017/")
    return client


def plotHistogram(lengths: ArrayLike, title=None) -> None:
    # sizeUniqueValues = len(np.unique(lengths.astype(int)))
    plt.hist(lengths, 50, edgecolor="black")
    if title is None:
        plt.title("Histogram over article length")
    plt.xlabel("Article Length")
    plt.ylabel("Frequency")

    mean = np.mean(lengths)
    std_dev = np.std(lengths)

    plt.axvline(mean, color="r", linestyle="dashed", linewidth="2", label="Mean")
    plt.axvline(mean + std_dev, color="g", linestyle="dashed", linewidth="2", label="Mean + std")
    plt.axvline(mean - std_dev, color="g", linestyle="dashed", linewidth="2", label="Mean - std")

    plt.legend()
    textstr = f"Mean: {mean:.2f}\nStandard Deviation: {std_dev:.2f}"
    props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
    plt.gca().text(
        0.05, -0.20, textstr, transform=plt.gca().transAxes, fontsize=15, verticalalignment="top", bbox=props
    )
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Connect to database
    client = connect()
    # db = client["nytimes"] # To connect to the full database
    db = client["nytimes_sample"]
    article_table = db["articles"]
    result = article_table.find().limit(100)
    assert result is not None
    print(len(result))

    # print(result["byline"])
# byline.person.firstname / .lastname /.title
