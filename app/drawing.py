import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


sns.set(style="whitegrid", palette="pastel", font_scale=1.2)

# — Функция для гистограммы сообщений по дням —
def draw_histogram(df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 6))
    daily_counts = df['date'].value_counts().sort_index()
    sns.lineplot(x=daily_counts.index, y=daily_counts.values, ax=ax)
    ax.set_title('Распределение сообщений по дням')
    ax.set_xlabel('Дата')
    ax.set_ylabel('Количество сообщений')
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig

# — Функция для распределения тональностей по дням недели —
def draw_sentiment_weekday(df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 6))
    df['weekday'] = df['date'].dt.day_name()
    sns.countplot(data=df, x='weekday', hue='sentiment', order=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'], ax=ax)
    ax.set_title('Тональности по дням недели')
    ax.set_xlabel('День недели')
    ax.set_ylabel('Количество сообщений')
    plt.xticks(rotation=30)
    plt.tight_layout()
    return fig

# — Функция для распределения тональностей по часам —
def draw_sentiment_hour(df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 6))
    df['hour'] = df['date'].dt.hour
    sns.countplot(data=df, x='hour', hue='sentiment', ax=ax)
    ax.set_title('Тональности по часам')
    ax.set_xlabel('Час суток')
    ax.set_ylabel('Количество сообщений')
    plt.tight_layout()
    return fig

# — Функция для топ-слов —
def draw_top_words(df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 6))
    words = pd.Series(' '.join(df['text']).lower().split()).value_counts().head(20)
    sns.barplot(x=words.values, y=words.index, ax=ax, orient='h')
    ax.set_title('Топ-20 слов')
    ax.set_xlabel('Количество упоминаний')
    plt.tight_layout()
    return fig

# — Функция для топ-эмодзи —
def draw_top_emoji(df: pd.DataFrame) -> plt.Figure:
    import emoji
    fig, ax = plt.subplots(figsize=(8, 6))
    all_emojis = ''.join(c for c in ' '.join(df['text']) if c in emoji.EMOJI_DATA)
    emoji_counts = pd.Series(list(all_emojis)).value_counts().head(10)
    sns.barplot(x=emoji_counts.values, y=emoji_counts.index, ax=ax, orient='h')
    ax.set_title('Топ-10 эмодзи')
    ax.set_xlabel('Количество')
    plt.tight_layout()
    return fig
