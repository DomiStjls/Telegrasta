from fastapi import FastAPI, Request, UploadFile, File, Form, Query
from typing import List, Optional
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse, FileResponse

import pandas as pd
import matplotlib.pyplot as plt
import os
import json
from app.utils import (
    preprocess_chat_data,
    plot_image,
    general_stats,
    user_stats,
    generate_wordclouds,
)
from fastapi.responses import StreamingResponse
import io

from fpdf import FPDF
from app.drawing import (
    draw_histogram,
    draw_sentiment_weekday,
    draw_sentiment_hour,
    draw_top_words,
)
import uuid

app = FastAPI()
templates = Jinja2Templates(directory="templates")
os.makedirs("data", exist_ok=True)
os.makedirs("static", exist_ok=True)
# Для статики (CSS, картинки и т.п.)
app.mount("/static", StaticFiles(directory="static"), name="static")
data_dir = "data"
key = None
# Глобальные переменные
chat_df = pd.DataFrame()
# Путь до текущего выбранного csv

favicon_path = 'favicon.ico'
@app.get('/favicon.ico', include_in_schema=False)
async def favicon():
    return FileResponse(favicon_path)

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    # Проверяем, есть ли уже загруженный CSV
    return templates.TemplateResponse("upload.html", {"request": request})


@app.post("/load", response_class=HTMLResponse)
async def load_by_key(request: Request, key: str = Form(...)):
    global chat_df
    try:
        c = 0
        for filename in os.listdir("./data/" + key):
            if filename.startswith("filtered_df"):
                c += 1

            chat_df = pd.read_csv(f"./data/{key}/filtered_df{c}.csv")
            chat_df["date"] = pd.to_datetime(chat_df["date"])
            chat_df["day"] = pd.to_datetime(chat_df["day"])
            chat_df["day"] = chat_df["day"].dt.date
            chat_df["weekday"] = chat_df["date"].dt.day_name()
            chat_df["hour"] = chat_df["date"].dt.hour
    except Exception as e:
        print(e)
        return RedirectResponse(url="/", status_code=303)

    return RedirectResponse(url=f"/statistics/{key}", status_code=303)


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    global chat_df, name

    contents = await file.read()

    # Декодируем байты в текст и парсим как JSON
    data = json.loads(contents.decode("utf-8"))["messages"]
    key = str(uuid.uuid4())
    user_dir = os.path.join(data_dir, key)
    static_dir = os.path.join("static", key)
    os.makedirs(user_dir)
    os.makedirs(static_dir)
    df = pd.DataFrame(data)
    chat_df, name = preprocess_chat_data(df, key)
    chat_df.to_csv(name, index=False)
    start = chat_df["date"].min().date()
    end = chat_df["date"].max().date()

    return RedirectResponse(
        url=f"/statistics/{key}?start={start}&end={end}", status_code=303
    )


@app.get("/statistics/{key}", response_class=HTMLResponse)
async def statistics_page(
    request: Request,
    key: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
    sentiments_new: Optional[List[str]] = Query(default=[]),
    authors_new: Optional[List[str]] = Query(default=[]),
):
    global chat_df
    if chat_df.empty:
        try:
            c = 0
            for filename in os.listdir("./data/" + key):
                if filename.startswith("filtered_df"):
                    c += 1

                chat_df = pd.read_csv(f"./data/{key}/filtered_df{c}.csv")
                chat_df["date"] = pd.to_datetime(chat_df["date"])
                chat_df["day"] = pd.to_datetime(chat_df["day"])
                chat_df["day"] = chat_df["day"].dt.date
                chat_df["weekday"] = chat_df["date"].dt.day_name()
                chat_df["hour"] = chat_df["date"].dt.hour
        except Exception as e:
            print(e)
            return RedirectResponse(url="/", status_code=303)
    if start is None:
        start = chat_df["date"].min().date()
    if end is None:
        end = chat_df["date"].max().date()

    if len(sentiments_new):
        sentiments = sentiments_new
    else:
        sentiments = chat_df["sentimental"].unique().tolist()

    if len(authors_new):
        authors = authors_new
    else:
        authors = chat_df["from"].unique().tolist()
    if chat_df.empty:
        return RedirectResponse("/", status_code=303)
    static_path = f"./static/{key}"
    folder = static_path
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            os.unlink(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))
    histogram = plot_image("histogram", start, end, sentiments, chat_df, key)
    sentiment_weekday = plot_image(
        "sentiment_weekday", start, end, sentiments, chat_df, key
    )
    sentiment_hour = plot_image("sentiment_hour", start, end, sentiments, chat_df, key)
    top_words = plot_image("top_words", start, end, sentiments, chat_df, key)
    top_emoji = plot_image("top_emoji", start, end, sentiments, chat_df, key)
    general = general_stats(chat_df, start, end)
    user_table = user_stats(chat_df, start, end)
    top_words_positive = plot_image("top_words_positive", start, end, sentiments, chat_df, key)
    top_words_negative = plot_image("top_words_negative", start, end, sentiments, chat_df, key)
    top_words_neutral = plot_image("top_words_neutral", start, end, sentiments, chat_df, key)
    generate_wordclouds(chat_df, key, start, end)
    # print(sentiments)
    return templates.TemplateResponse(
        "statistics.html",
        {
            "key": key,
            "request": request,
            "all_sentiments": chat_df["sentimental"].unique().tolist(),
            "sentiments": sentiments,
            "all_authors": chat_df["from"].unique().tolist(),
            "authors": authors,
            "start_date": start,
            "end_date": end,
            "histogram": histogram,
            "sentiment_weekday": sentiment_weekday,
            "sentiment_hour": sentiment_hour,
            "top_words": top_words,
            "top_emoji": top_emoji,
            "general": general,
            "user_table": user_table.to_dict(orient="records"),
            "top_words_positive": top_words_positive,
            "top_words_negative": top_words_negative,
            "top_words_neutral": top_words_neutral,


        },
    )


# Генерация PDF отчета
@app.get("/download_report")
async def download_report():
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    def add_plot_to_pdf(plot_func, title):
        buf = io.BytesIO()
        fig = plot_func(chat_df)
        fig.savefig(buf, format="png", bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        pdf.add_page()
        pdf.set_font("Arial", size=16)
        pdf.cell(0, 10, title, ln=True, align="C")
        pdf.image(buf, x=10, y=30, w=pdf.w - 20)
        buf.close()

    # Добавляем нужные графики
    add_plot_to_pdf(draw_histogram, "Распределение сообщений по дням")
    add_plot_to_pdf(draw_sentiment_weekday, "Тональности по дням недели")
    add_plot_to_pdf(draw_sentiment_hour, "Тональности по времени суток")
    add_plot_to_pdf(draw_top_words, "Частые слова")
    # add_plot_to_pdf(draw_top_emoji, "Топ эмодзи")
    

    output = io.BytesIO()
    pdf.output(output)
    output.seek(0)

    return StreamingResponse(
        output,
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=chat_report.pdf"},
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
