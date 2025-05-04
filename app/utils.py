from transformers import pipeline
import pandas as pd
from pymorphy3 import MorphAnalyzer
import os
import emoji
import numpy as np
from collections import Counter
from wordcloud import WordCloud
from io import BytesIO
import base64
import seaborn as sns
import uuid
import re
from matplotlib import pyplot as plt

model = pipeline(model="seara/rubert-tiny2-russian-sentiment")
morph = MorphAnalyzer()
# plt.rcParams['font.family'] = 'Segoe UI Emoji'


def model_analysis(text):
    return model(text)[0]["label"]


def filter_words(words):
    filtered_words = []
    for word in words:
        parsed_word = morph.parse(word)[0]
        if parsed_word.tag.POS in {
            "NOUN",
            "ADJF",
            "ADJS",
            "INFN",
            "VERB",
        }:  # если это существительное, прилагательное или глагол
            filtered_words.append(word)
    processed_text = " ".join(filtered_words)

    return processed_text


def preprocess_chat_data(df, key):
    df = df[
        [
            "id",
            "type",
            "date",
            "date_unixtime",
            "from",
            "from_id",
            "text",
        ]  # базовый вариант без фотографий и всяких излишеств
    ]
    filtered_df = df[df["text"].apply(lambda x: isinstance(x, str) and len(x) > 0)]

    # n_all = len(filtered_df["text"])
    # authors = filtered_df["from"].unique()

    sent_analysis = filtered_df["text"].apply(model_analysis)
    filtered_df["sentimental"] = sent_analysis
    filtered_df["date"] = pd.to_datetime(filtered_df["date"])
    filtered_df["day"] = filtered_df["date"].dt.date
    filtered_df["weekday"] = filtered_df["date"].dt.day_name()
    filtered_df["hour"] = filtered_df["date"].dt.hour
    c = 0
    for filename in os.listdir("./data/" + key):
        if filename.startswith("filtered_df"):
            c += 1
    name = f"./data/{key}/filtered_df{c + 1}.csv"
    # filtered_df.to_csv(name, index=False)  # "../data/filtered_df3.csv"
    return filtered_df, name


def sentiment_profile_by_hour(df):
    grouped = df.groupby(["hour", "sentimental"]).size().reset_index(name="count")
    total = grouped.groupby("hour")["count"].transform("sum")
    grouped["percent"] = grouped["count"] / total * 100
    return grouped


def sentiment_by_weekday_ratio(df):
    grouped = df.groupby(["weekday", "sentimental"]).size().reset_index(name="count")
    total = grouped.groupby("weekday")["count"].transform("sum")
    grouped["percent"] = grouped["count"] / total * 100
    return grouped


def generate_wordcloud(text):
    wc = WordCloud(
        width=800, height=400, background_color="white", max_words=100
    ).generate(text)
    buffer = BytesIO()
    wc.to_image().save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/png;base64,{encoded}"


def most_common_emojis(text_series):
    emojis = list(
        "".join(
            [
                "".join(emoji.distinct_emoji_list(c))
                for c in text_series.dropna().astype(str)
            ]
        )
    )
    emoji_freq = Counter(emojis).most_common(3)
    return " ".join([em for em, count in emoji_freq])


def extract_top_words(df):
    all_words = []
    for text in df["text"].dropna():
        words = re.findall(r"\b\w+\b", text.lower())
        all_words.extend(words)
    return pd.Series(Counter(all_words)).sort_values(ascending=False)


def extract_top_emoji(df):
    all_emojis = []
    for text in df["text"].dropna():
        all_emojis.extend([ch for ch in text if ch in emoji.EMOJI_DATA])
    return pd.Series(Counter(all_emojis)).sort_values(ascending=False)


def plot_image(
    plot_type: str,
    start: str = None,
    end: str = None,
    sentiments: list = [],
    chat_df=None,
    key: str = None,
):
    df_filtered = chat_df.copy()

    if start and end:
        df_filtered = df_filtered[
            (pd.to_datetime(df_filtered["date"]) >= pd.to_datetime(start))
            & (pd.to_datetime(df_filtered["date"]) <= pd.to_datetime(end))
        ]
    if len(sentiments):
        # sentiments_list = sentiments.split(",")
        df_filtered = df_filtered[df_filtered["sentimental"].isin(sentiments)]

    fig, ax = plt.subplots(figsize=(12, 4))

    plt.rc("axes", unicode_minus=False)
    sns.set_theme()

    if plot_type == "histogram":
        sns.histplot(
            data=df_filtered,
            x="day",
            hue="day",
            palette="Blues_d",
            legend=False,
            bins=30,
            ax=ax,
        )
        ax.set_title("Распределение сообщений по дням")

    elif plot_type == "sentiment_weekday":
        weekday_df = sentiment_by_weekday_ratio(
            df_filtered
        )  # должен возвращать: weekday, sentimental, percent
        sns.barplot(
            data=weekday_df,
            x="weekday",
            y="percent",
            palette="Blues_d",
            hue="sentimental",
            ax=ax,
        )
        ax.set_title("percent сообщений каждой тональности по дням недели")

    elif plot_type == "sentiment_hour":
        hour_df = sentiment_profile_by_hour(
            df_filtered
        )  # должен возвращать: hour, sentimental, percent
        sns.lineplot(
            data=hour_df,
            x="hour",
            y="percent",
            hue="sentimental",
            palette="Blues_d",
            marker="o",
            ax=ax,
        )
        ax.set_title("percent сообщений по времени суток")
    elif plot_type == "top_words":
        top_words_series = extract_top_words(df_filtered)
        sns.barplot(
            y=top_words_series.index[:10],
            x=top_words_series.values[:10],
            hue=top_words_series.index[:10],
            palette="Blues_d",
            ax=ax,
        )
        ax.set_title("Топ-слова")
        ax.set_xlabel("Частота")
        ax.set_ylabel("Слова")

    elif plot_type == "top_emoji":
        top_emoji_series = extract_top_emoji(df_filtered)
        em_df = pd.DataFrame(
            {"emoji": top_emoji_series.index, "count": top_emoji_series.values}
        )
        # table = "\n".join(
        #     [f"<tr>\n<td>{r['emoji']}</td>\n<td>{r['count']}</td>\n</tr>" for i, r in em_df.iterrows()]
        # )
        # table = {"emoji": top_emoji_series.index[:10].tolist(), "count": top_emoji_series.values[:10].tolist()
        # }
        a_emoji = " ".join(
            [r['emoji'] for i, r in em_df.iterrows()][:5]
        )
        return a_emoji

    else:
        plt.text(0.5, 0.5, "Ups", ha="center", va="center")

    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    # img_base64 = base64.b64encode(buf.read()).decode("utf-8")

    unique_id = str(uuid.uuid4())  # Уникальное имя файла
    filename = f"{plot_type}_{unique_id}.png"
    filepath = os.path.join(f"static/{key}", filename)

    fig.savefig(filepath, bbox_inches="tight")
    plt.close(fig)

    return f"/static/{key}/{filename}"


def general_stats(df_n: pd.DataFrame, start, end) -> dict:
    df = df_n.copy()
    if start and end:
        df = df[
            (pd.to_datetime(df["date"]) >= pd.to_datetime(start))
            & (pd.to_datetime(df["date"]) <= pd.to_datetime(end))
        ]

    total_messages = len(df)
    authors = df["from"].unique().tolist()

    sentiment_counts = df["sentimental"].value_counts(normalize=True) * 100
    sentiment_percent = {
        "positive": round(sentiment_counts.get("positive", 0), 2),
        "neutral": round(sentiment_counts.get("neutral", 0), 2),
        "negative": round(sentiment_counts.get("negative", 0), 2),
    }

    first_messages = df.sort_values("date").groupby("day").first()
    first_speaker = first_messages["from"].value_counts().idxmax()

    # Вычисляем час, если не был заранее рассчитан
    if "hour" not in df.columns:
        df["hour"] = pd.to_datetime(df["date"]).dt.hour

    peak_hour = df["hour"].value_counts().idxmax()

    return {
        "total_messages": total_messages,
        "authors": authors,
        "sentiment_percent": sentiment_percent,
        "first_speaker": first_speaker,
        "peak_hour": peak_hour,
    }


def user_stats(df_n: pd.DataFrame, start, end) -> pd.DataFrame:
    df = df_n.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["day"] = pd.to_datetime(df["day"])

    # Фильтрация по дате
    if start and end:
        df = df[
            (df["date"] >= pd.to_datetime(start)) & (df["date"] <= pd.to_datetime(end))
        ]

    df["hour"] = df["date"].dt.hour
    grouped = df.groupby("from")

    # Считаем тексты короткого и длинного сообщений
    shortest_texts = grouped.apply(
        lambda g: g.loc[g["text"].str.len().idxmin()]["text"] if not g.empty else ""
    )
    longest_texts = grouped.apply(
        lambda g: g.loc[g["text"].str.len().idxmax()]["text"] if not g.empty else ""
    )

    # Базовая статистика
    stats = grouped["text"].agg(
        [
            ("total_messages", "count"),
            ("avg_length", lambda x: x.str.len().mean()),
        ]
    )

    # Среднее число сообщений в день
    days_active = grouped["day"].nunique()
    stats["avg_per_day"] = stats["total_messages"] / days_active

    # Проценты по тональности
    sentiment_ratio = df.groupby(["from", "sentimental"]).size().unstack(fill_value=0)
    sentiment_percent = sentiment_ratio.div(sentiment_ratio.sum(axis=1), axis=0) * 100

    stats = stats.join(sentiment_percent, how="left").fillna(0)

    # Пиковый час активности
    peak_hours = grouped["hour"].agg(
        lambda x: x.value_counts().idxmax() if not x.empty else None
    )
    stats["peak_hour"] = peak_hours

    # Добавим тексты сообщений
    stats["shortest_msg"] = shortest_texts
    stats["longest_msg"] = longest_texts
    
    return stats.reset_index()
