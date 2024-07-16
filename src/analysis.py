import numpy as np
from matplotlib import pyplot as plt
from numpy.typing import ArrayLike
from pymongo import MongoClient
from pymongo.cursor import Cursor
from scipy.stats import gamma, normaltest
from tqdm import tqdm
import os
import json


def connect():
    client = MongoClient("mongodb://root:secure_pw@localhost:27017/")
    return client


def plotHistogram(lengths: ArrayLike, target: str, title=None) -> None:
    lengths = lengths if isinstance(lengths, np.ndarray) else np.array(lengths)
    sizeUniqueValues = len(np.unique(lengths.astype(int)))
    mean = np.mean(lengths)
    std_dev = np.std(lengths)
    limit = 300 if target == "caption" else 4500

    shape, loc, scale = gamma.fit(lengths, loc=0, scale=20)
    x = np.linspace(0, np.max(lengths), 200)
    plt.plot(
        x,
        gamma.pdf(x, shape, loc, scale),
        "g-",
        lw=4,
        alpha=0.8,
        label="MLE gamma function",
    )

    plt.hist(lengths[lengths < 300], 150, edgecolor="black", density=True)
    if title is None:
        title = f"Histogram over length"
    plt.title(title)
    plt.xlim(0, limit)
    plt.xlabel(f"{target} Length")
    plt.ylabel("Frequency")

    plt.axvline(mean, color="r", linestyle="dashed", linewidth="2", label="Mean")
    plt.axvline(mean + std_dev, color="g", linestyle="dashed", linewidth="2", label="Mean + std")
    plt.axvline(mean - std_dev, color="g", linestyle="dashed", linewidth="2", label="Mean - std")

    plt.legend()
    p_value = normaltest(lengths)
    textstr = f"Mean: {mean:.2f}\nStandard Deviation: {std_dev:.2f}\np-value for normal dist.: {p_value[1]:.2f}"
    textDistr = f"Alpha: {shape:.2f}\nLoc:{loc:.2f}\nScale: {scale:.2f}"
    props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
    propsDistr = dict(boxstyle="round", facecolor="green", alpha=0.5)
    plt.gca().text(
        0.05,
        -0.20,
        textstr,
        transform=plt.gca().transAxes,
        fontsize=15,
        verticalalignment="top",
        bbox=props,
    )
    plt.gca().text(
        0.65,
        -0.20,
        textDistr,
        transform=plt.gca().transAxes,
        fontsize=15,
        verticalalignment="top",
        bbox=propsDistr,
    )
    plt.tight_layout()
    analysis_dir = os.path.join(os.getcwd(), "analysis")
    os.makedirs(analysis_dir, exist_ok=True)
    plt.savefig(f"{analysis_dir}/analysis_{target}_{title}.png")
    plt.close()

    with open(f"{analysis_dir}/analysis_{target}_{title}.json", "w") as f:
        json.dump({
            "mean": mean,
            "std_dev": std_dev,
            "normal_test_statistic": p_value[0],
            "p_value": p_value[1],
            "alpha": shape,
            "loc": loc,
            "scale": scale,
        },
            f
        )


def countLengthArticle(article: Cursor, target: str) -> int:
    length = 0
    if target == "text":
        for section in article["parsed_section"]:
            if section["type"] == "paragraph":
                # Alternatively use nltk.TweetTokenizer, but beware punctuation
                length += str(section["text"]).count(" ") + 1
    elif target == "caption":
        for section in article["parsed_section"]:
            if section["type"] == "caption":
                length += str(section["text"]).count(" ") + 1
    return length


def run(article_table, target):
    # Preprocess all articles
    preprocessed_articles = []
    lengths = []
    fullNames = []
    for article in tqdm(article_table.find().limit(0), total=450_000, desc="Analyzing articles"):
        if article.get("byline", None) is None:
            continue
        if isinstance(article["byline"], list):
            continue
        if article["byline"].get("person", None) is None:
            continue
        authorInfo = article["byline"]["person"]
        if len(authorInfo) == 0:
            continue
        authorInfo = authorInfo[0]
        fullName = (
            authorInfo["firstname"] + authorInfo["lastname"]
            if "lastname" in authorInfo.keys() and authorInfo["lastname"] is not None
            else ""
        )

        length = countLengthArticle(article, target)
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

    plotHistogram(authorLengths, target, title="Average length per author")


if __name__ == "__main__":
    # Connect to database
    client = connect()
    db = client["nytimes"]
    article_table = db["articles"]

    run(article_table, "caption")
    run(article_table, "text")
