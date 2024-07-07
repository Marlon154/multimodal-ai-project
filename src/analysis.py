from pymongo import MongoClient, collection
from pymongo.cursor import Cursor
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from typing import List, Iterable, Union
from numpy.typing import ArrayLike
from scipy.stats import gamma


def connect():
    client = MongoClient("mongodb://root:secure_pw@localhost:27017/")
    return client


def plotHistogram(lengths: ArrayLike, target, title=None) -> None:
    lengths = lengths if isinstance(lengths, np.ndarray) else np.array(lengths)
    sizeUniqueValues = len(np.unique(lengths.astype(int)))

    # shape, loc, scale = gamma.fit(lengths, loc=-80, scale=200)
    shape, loc, scale = gamma.fit(lengths)
    x = np.linspace(0, np.max(lengths), 200)
    print(f"{(shape, loc, scale)=}")
    plt.plot(x, gamma.pdf(x, shape, loc, scale), "r-", lw=4, alpha=0.6, label="gamma pdf")

    plt.hist(lengths, 150, edgecolor="black", density=True)
    if title is None:
        plt.title(f"Histogram over article length")
    plt.xlabel(f"{target} Length")
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
    plt.savefig(f"./analysis/analysis_{target}_{title}.png")
    plt.show()



def countLengthArticle(article: Cursor, target: str = "caption") -> int:
    length = 0
    if target == "text":
        for section in article["parsed_section"]:
            if section["type"] == "paragraph":
                # Alternatively use nltk.TweetTokenizer, but beware punctuation
                length += str(section["text"]).count(" ") + 1
    elif target == "caption":
        for section in article["parsed_section"]:
            if section["type"] == "caption":
                # Alternatively use nltk.TweetTokenizer, but beware punctuation
                length += str(section["text"]).count(" ") + 1
    return length


if __name__ == "__main__":
    target = "text"
    # Connect to database
    client = connect()
    db = client["nytimes"] # To connect to the full database
    # db = client["nytimes_sample"]
    article_table = db["articles"]

    # Preprocess all articles
    preprocessed_articles = []
    lengths = []
    fullNames = []
    for article in tqdm(article_table.find().limit(10000)):
        if article.get("byline", None) is None:
            continue
        if article["byline"].get("person", None) is None:
            continue
        authorInfo = article["byline"]["person"]
        if len(authorInfo) == 0:
            continue
        authorInfo = authorInfo[0]
        fullName = authorInfo["firstname"] + authorInfo["lastname"] if "lastname" in authorInfo.keys() and authorInfo["lastname"] is not None else ""

        length = countLengthArticle(article)
        lengths.append(length)

        fullNames.append(fullName)
        preprocessed_articles.append((length, authorInfo, fullName))

    plotHistogram(lengths, target)

    nameMarginalized = {name: [] for name in fullNames}
    for articleInfo in preprocessed_articles:
        nameMarginalized[articleInfo[2]].append(articleInfo[0])

    authorLengths = []
    for authorLength in nameMarginalized.values():
        authorLengths.append(sum(authorLength) / len(authorLength))

    plotHistogram(authorLengths, target, title="Average article length per author")

    # print(result["byline"])
# byline.person.firstname / .lastname /.title
