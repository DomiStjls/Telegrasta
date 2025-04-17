import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import numpy as np
from collections import Counter
from wordcloud import WordCloud
import base64
from io import BytesIO
import emoji

# Загрузка CSV
chat_df = pd.read_csv("./data/filtered_df3.csv")
chat_df['date'] = pd.to_datetime(chat_df['date'], format="%Y-%m-%dT%H:%M:%S")
chat_df['day'] = chat_df['date'].dt.date
chat_df['weekday'] = chat_df['date'].dt.day_name()
chat_df['hour'] = chat_df['date'].dt.hour

# Получение участников
authors = chat_df['from'].unique()
sentiments = chat_df['sentimental'].unique().tolist()

# Расчёт долей по тональности по дням недели

def sentiment_by_weekday_ratio(df):
    sentiment_counts = df['sentimental'].value_counts()
    weekday_sentiment = df.groupby(['weekday', 'sentimental']).size().unstack(fill_value=0)
    for sentiment in weekday_sentiment.columns:
        total = sentiment_counts.get(sentiment, 1)
        weekday_sentiment[sentiment] = weekday_sentiment[sentiment] / total * 100
    weekday_sentiment = weekday_sentiment.reset_index().melt(id_vars='weekday', var_name='Тональность', value_name='Процент')
    return weekday_sentiment

# Эмоциональный профиль по времени суток (проценты)

def sentiment_profile_by_hour(df):
    sentiment_counts = df['sentimental'].value_counts()
    hourly_sentiment = df.groupby(['hour', 'sentimental']).size().unstack(fill_value=0)
    for sentiment in hourly_sentiment.columns:
        total = sentiment_counts.get(sentiment, 1)
        hourly_sentiment[sentiment] = hourly_sentiment[sentiment] / total * 100
    hourly_sentiment = hourly_sentiment.reset_index().melt(id_vars='hour', var_name='Тональность', value_name='Процент')
    return hourly_sentiment

# Создание облака слов

def generate_wordcloud(text):
    wc = WordCloud(width=800, height=400, background_color='white', max_words=100).generate(text)
    buffer = BytesIO()
    wc.to_image().save(buffer, format='PNG')
    encoded = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/png;base64,{encoded}"

# Поиск часто используемых эмодзи

def most_common_emojis(text_series):
    emojis = list(''.join([''.join(emoji.distinct_emoji_list(c))for c in text_series.dropna().astype(str)]))
    emoji_freq = Counter(emojis).most_common(3)
    return " ".join([em for em, count in emoji_freq])

# Dash-приложение
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Анализ Telegram-чата"

# Интерфейс
app.layout = dbc.Container([
    dcc.Loading(
        type="circle",
        fullscreen=True,
        children=html.Div([
            html.H1("Анализ Telegram-чата", className="my-4"),

            html.P("Этот сайт анализирует диалог из Telegram между двумя участниками. Загруженный CSV-файл содержит историю переписки с тональностью сообщений. Здесь ты можешь увидеть интересную статистику и графики!"),

            dbc.Row([
                dbc.Col([
                    html.Label("Выбери диапазон дат:"),
                    dcc.DatePickerRange(
                        id='date-range',
                        min_date_allowed=chat_df['date'].min().date(),
                        max_date_allowed=chat_df['date'].max().date(),
                        start_date=chat_df['date'].min().date(),
                        end_date=chat_df['date'].max().date(),
                        display_format='DD.MM.YYYY'
                    )
                ], width=6),

                dbc.Col([
                    html.Label("Фильтр по тональности:"),
                    dcc.Dropdown(
                        id='sentiment-filter',
                        options=[{'label': s, 'value': s} for s in sentiments],
                        value=sentiments,
                        multi=True
                    )
                ], width=6)
            ], className="mb-4"),

            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H5("Общее количество сообщений", className="card-title"),
                            html.Div(id="total-messages")
                        ])
                    ])
                ], width=3),

                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H5("Диалог с", className="card-title"),
                            html.Div(id="dialog-period")
                        ])
                    ])
                ], width=3),

                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H5("Наиболее активное время суток", className="card-title"),
                            html.Div(id="peak-time")
                        ])
                    ])
                ], width=3),

                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H5("Общее настроение", className="card-title"),
                            html.Div(id="overall-sentiment")
                        ])
                    ])
                ], width=3)
            ], className="mb-4"),

            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H5("Кто чаще инициирует диалог", className="card-title"),
                            html.Div(id="initiator")
                        ])
                    ])
                ], width=3),
            ], className="mb-4"),

            dcc.Graph(id="histogram"),
            dcc.Graph(id="sentiment-weekday-ratio"),
            dcc.Graph(id="sentiment-hour-profile"),

            html.Hr(),
            html.H2("Статистика по каждому пользователю", className="mt-5"),

            html.Div(id="user-metrics")
        ])
    )
], fluid=True)
@app.callback(
    Output("total-messages", "children"),
    Output("dialog-period", "children"),
    Output("peak-time", "children"),
    Output("overall-sentiment", "children"),
    Output("histogram", "figure"),
    Output("initiator", "children"),
    Output("sentiment-weekday-ratio", "figure"),
    Output("sentiment-hour-profile", "figure"),
    Input("date-range", "start_date"),
    Input("date-range", "end_date"),
    Input("sentiment-filter", "value")
)
def update_overall_metrics(start_date, end_date, selected_sent):
    df_filtered = chat_df[
        (chat_df['date'] >= start_date) & 
        (chat_df['date'] <= end_date) &
        (chat_df['sentimental'].isin(selected_sent))
    ]

    total_msgs = len(df_filtered)
    first_msg_time = df_filtered['date'].min().strftime("%d.%m.%Y") if total_msgs > 0 else "-"
    last_msg_time = df_filtered['date'].max().strftime("%d.%m.%Y") if total_msgs > 0 else "-"
    most_active_time = df_filtered['date'].dt.hour.mode()[0] if total_msgs > 0 else "-"
    sentiment_counts = df_filtered['sentimental'].value_counts(normalize=True).to_dict()
    sentiment_text = " ".join([f"{k}: {v:.0%}" for k, v in sentiment_counts.items()]) if total_msgs > 0 else "-"

    hist_fig = px.histogram(df_filtered, x="day", nbins=30, title="Распределение сообщений по дням")
    hist_fig.update_layout(xaxis_title="Дата", yaxis_title="Количество сообщений")

    weekday_ratio_df = sentiment_by_weekday_ratio(df_filtered)
    weekday_fig = px.bar(weekday_ratio_df, x='weekday', y='Процент', color='Тональность', barmode='group', title="Процент сообщений каждой тональности по дням недели")
    weekday_fig.update_layout(xaxis_title="День недели", yaxis_title="Процент (%)")

    hour_profile_df = sentiment_profile_by_hour(df_filtered)
    hour_fig = px.line(hour_profile_df, x='hour', y='Процент', color='Тональность', markers=True, title="Процент сообщений каждой тональности по времени суток")
    hour_fig.update_layout(xaxis_title="Час", yaxis_title="Процент (%)")

    initiators = df_filtered.sort_values('date').groupby('day').first()['from'].value_counts()
    initiator_text = ", ".join([f"{k}: {v}" for k, v in initiators.items()])

    return (
        f"{total_msgs}",
        f"{first_msg_time} по {last_msg_time}",
        f"{most_active_time}:00" if most_active_time != "-" else "-",
        sentiment_text,
        hist_fig,
        initiator_text,
        weekday_fig,
        hour_fig
    )
@app.callback(
    Output("user-metrics", "children"),
    Input("date-range", "start_date"),
    Input("date-range", "end_date"),
    Input("sentiment-filter", "value")
)
def update_user_metrics(start_date, end_date, selected_sent):
    cards = []
    for author in authors:
        user_df = chat_df[
            (chat_df['from'] == author) & 
            (chat_df['date'] >= start_date) & 
            (chat_df['date'] <= end_date) &
            (chat_df['sentimental'].isin(selected_sent))
        ]

        total = len(user_df)
        avg_len = user_df['text'].astype(str).apply(len).mean() if total > 0 else 0
        sentiment_counts = user_df['sentimental'].value_counts(normalize=True).to_dict()

        texts = user_df['text'].dropna().astype(str)
        words = texts.str.cat(sep=' ').lower().split()
        words = [w for w in words if len(w) > 3]

        emojis = list(''.join([''.join(emoji.distinct_emoji_list(c))for c in texts]))
        emoji_freq = Counter(emojis).most_common(3)
        emoji_text = " ".join([em for em, count in emoji_freq])

        laugh_count = texts.str.count(r"а{2,}|х{2,}|л{2,}").sum()
        transitions = user_df['sentimental'].shift() != user_df['sentimental']
        sentiment_changes = transitions.sum()

        longest = user_df.loc[user_df['text'].astype(str).apply(len).idxmax()] if total > 0 else None
        shortest = user_df.loc[user_df['text'].astype(str).apply(len).idxmin()] if total > 0 else None
        questions = texts[texts.str.endswith("?")].tolist()
        questions_often = [k for k, v in Counter(questions).most_common(3)]
        unique_words = set(words)

        # WordCloud generation
        wordcloud_img = None
        if words:
            wordcloud = WordCloud(width=500, height=500, background_color='white').generate(' '.join(words))
            buf = BytesIO()
            wordcloud.to_image().save(buf, format='PNG')
            encoded = base64.b64encode(buf.getvalue()).decode()
            wordcloud_img = html.Div([
                html.H6("Облако слов", className="text-center text-muted mb-2"),
                html.Img(src=f'data:image/png;base64,{encoded}', style={
                    "width": "100%",
                    "height": "auto",
                    "maxWidth": "500px",
                    "maxHeight": "500px",
                    "border": "1px solid #ccc",
                    "padding": "0.5rem",
                    "borderRadius": "0.5rem",
                    "backgroundColor": "#f8f9fa"
                })
            ])

        # Sentiment Progress Bars
        sentiment_bars = []
        for k in ['positive', 'neutral', 'negative']:
            if k in sentiment_counts:
                sentiment_bars.append(
                    dbc.Progress(
                        value=round(sentiment_counts[k]*100),
                        color={"positive": "success", "neutral": "secondary", "negative": "danger"}[k],
                        label=f"{k.capitalize()}: {sentiment_counts[k]*100:.0f}%",
                        striped=False,
                        animated=False,
                        className="mb-1",
                        style={"height": "1.5rem", "fontSize": "0.9rem"}
                    )
                )

        card = dbc.Card([
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.H4(f"Пользователь: {author}", className="card-title mb-3 text-primary"),
                        html.P(f"Всего сообщений: {total}"),
                        html.P(f"Средняя длина сообщения: {avg_len:.1f} символов"),
                        html.Div(sentiment_bars, className="mb-3"),
                        html.P(f"Смен настроения: {sentiment_changes}"),
                        html.P(f"Сообщений со смехом: {laugh_count}"),
                        html.P(f"Любимые смайлики: {emoji_text}" if emoji_text else "-"),
                        html.P(f"Самое длинное сообщение ({longest['date'].strftime('%d.%m.%Y')}):" if longest is not None else "-"),
                        html.Details([html.Summary("Показать"), html.P(longest['text'])]) if longest is not None else html.P("-"),
                        html.P(f"Самое короткое сообщение ({shortest['date'].strftime('%d.%m.%Y')}): {shortest['text']}" if shortest is not None else "-"),
                        html.P(f"Уникальных слов: {len(unique_words)}"),
                        html.P(" Частые вопросы: " + " || ".join(questions_often) if questions_often else "Нет вопросов")
                    ], width=8),
                    dbc.Col([
                        wordcloud_img
                    ], width=4)
                ])
            ])
        ], className="mb-4 shadow-sm border-info")

        cards.append(card)

    return html.Div(cards)
if __name__ == '__main__':
    app.run(debug=True)