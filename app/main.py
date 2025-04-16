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

# Загрузка CSV
chat_df = pd.read_csv("./data/filtered_df2.csv")
chat_df['date'] = pd.to_datetime(chat_df['date'], format="%Y-%m-%dT%H:%M:%S")
chat_df['day'] = chat_df['date'].dt.date
chat_df['weekday'] = chat_df['date'].dt.day_name()
chat_df['hour'] = chat_df['date'].dt.hour

# Получение участников
authors = chat_df['from'].unique()
sentiments = chat_df['sentimental'].unique().tolist()

# Dash-приложение
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Анализ Telegram-чата"

# Интерфейс
app.layout = dbc.Container([
    html.H1("Анализ Telegram-чата", className="my-4"),

    html.P("Привет! Этот сайт анализирует диалог из Telegram между двумя участниками. Загруженный CSV-файл содержит историю переписки с тональностью сообщений. Здесь ты можешь увидеть интересную статистику и графики!"),

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

    html.Hr(),
    html.H2("Статистика по каждому пользователю", className="mt-5"),

    dcc.Tabs(id="user-tabs", value=authors[0], children=[
        dcc.Tab(label=name, value=name) for name in authors
    ]),

    html.Div(id="user-metrics")
], fluid=True)

@app.callback(
    Output("total-messages", "children"),
    Output("dialog-period", "children"),
    Output("peak-time", "children"),
    Output("overall-sentiment", "children"),
    Output("histogram", "figure"),
    Output("initiator", "children"),
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

    hist_fig = px.histogram(df_filtered, x="date", nbins=30, title="Распределение сообщений по времени")
    hist_fig.update_layout(xaxis_title="Дата", yaxis_title="Количество сообщений")

    initiators = df_filtered.sort_values('date').groupby('day').first()['from'].value_counts()
    initiator_text = ", ".join([f"{k}: {v}" for k, v in initiators.items()])

    return (
        f"{total_msgs}",
        f"{first_msg_time} по {last_msg_time}",
        f"{most_active_time}:00" if most_active_time != "-" else "-",
        sentiment_text,
        hist_fig,
        initiator_text
    )

@app.callback(
    Output("user-metrics", "children"),
    Input("user-tabs", "value"),
    Input("date-range", "start_date"),
    Input("date-range", "end_date"),
    Input("sentiment-filter", "value")
)
def update_user_metrics(author, start_date, end_date, selected_sent):
    user_df = chat_df[
        (chat_df['from'] == author) & 
        (chat_df['date'] >= start_date) & 
        (chat_df['date'] <= end_date) &
        (chat_df['sentimental'].isin(selected_sent))
    ]

    total = len(user_df)
    avg_len = user_df['text'].astype(str).apply(len).mean() if total > 0 else 0
    sentiment_counts = user_df['sentimental'].value_counts(normalize=True).to_dict()

    words = user_df['text'].dropna().astype(str).str.cat(sep=' ').lower().split()
    words = [w for w in words if len(w) > 3]
    freq_words = pd.Series(words).value_counts().head(10)
    word_fig = px.bar(freq_words, x=freq_words.index, y=freq_words.values, labels={"x": "Слово", "y": "Частота"}, title="Частые слова")
    word_fig.update_layout(xaxis_title="Слово", yaxis_title="Частота")

    wordcloud = WordCloud(width=400, height=600, background_color='white').generate(" ".join(words))
    img = BytesIO()
    wordcloud.to_image().save(img, format='PNG')
    img.seek(0)
    encoded = base64.b64encode(img.read()).decode('utf-8')
    wordcloud_img = html.Img(src='data:image/png;base64,{}'.format(encoded), style={"maxWidth": "100%"})

    laugh_count = user_df['text'].str.count(r"а{2,}|х{2,}|л{2,}").sum()
    transitions = user_df['sentimental'].shift() != user_df['sentimental']
    sentiment_changes = transitions.sum()

    day_count = user_df.groupby("day").size().reset_index(name="count")
    timeline_fig = px.line(day_count, x="day", y="count", title="Сообщения по дням")
    timeline_fig.update_layout(xaxis_title="Дата", yaxis_title="Количество сообщений")

    weekday_sent = user_df.groupby(['weekday', 'sentimental']).size().reset_index(name='count')
    weekday_fig = px.bar(weekday_sent, x='weekday', y='count', color='sentimental', barmode='group', title="Настроения по дням недели")
    weekday_fig.update_layout(xaxis_title="День недели", yaxis_title="Количество сообщений")

    longest = user_df.loc[user_df['text'].astype(str).apply(len).idxmax()] if total > 0 else None
    shortest = user_df.loc[user_df['text'].astype(str).apply(len).idxmin()] if total > 0 else None

    questions = user_df[user_df['text'].astype(str).str.endswith("?")]['text'].tolist()[:3]
    unique_words = set(words)

    emotion_profile = user_df.groupby(['hour', 'sentimental']).size().reset_index(name='count')
    emotion_fig = px.line(emotion_profile, x='hour', y='count', color='sentimental', title='Эмоциональный профиль по времени суток')
    emotion_fig.update_layout(xaxis_title="Час", yaxis_title="Количество сообщений")

    return dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4(f"Пользователь: {author}", className="card-title mb-3"),
                    html.P(f"Всего сообщений: {total}"),
                    html.P(f"Средняя длина сообщения: {avg_len:.1f} символов"),
                    html.P("Тональность: " + ", ".join([f"{k}: {v:.0%}" for k, v in sentiment_counts.items()])),
                    html.P(f"Смен настроения: {sentiment_changes}"),
                    html.P(f"Сообщений со смехом: {laugh_count}"),
                    html.P(f"Самое длинное сообщение ({longest['date'].strftime('%d.%m.%Y')}):" if longest is not None else "-"),
                    html.Details([html.Summary("Показать"), html.P(longest['text'])]) if longest is not None else html.P("-"),
                    html.P(f"Самое короткое сообщение ({shortest['date'].strftime('%d.%m.%Y')}): {shortest['text']}" if shortest is not None else "-"),
                    html.P(f"Уникальных слов: {len(unique_words)}"),
                    html.P("Вопросы: " + " | ".join(questions) if questions else "Нет вопросов")
                ])
            ], className="mb-4")
        ], width=4),

        dbc.Col([
            dcc.Graph(figure=word_fig),
            wordcloud_img,
            dcc.Graph(figure=timeline_fig),
            dcc.Graph(figure=weekday_fig),
            dcc.Graph(figure=emotion_fig)
        ], width=8)
    ])

if __name__ == '__main__':
    app.run(debug=True)
