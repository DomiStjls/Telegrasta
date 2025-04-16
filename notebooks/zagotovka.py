# app/services/parser.py

import json
from typing import List, Dict, Any
from datetime import datetime


def load_telegram_json(path: str) -> Dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def extract_messages(json_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    raw_messages = json_data.get("messages", [])
    messages = []

    for msg in raw_messages:
        if msg["type"] != "message" or not msg.get("text"):
            continue

        if isinstance(msg["text"], list):
            # В некоторых json'ах text — это список (с форматированием)
            text = "".join(part if isinstance(part, str) else part.get("text", "") for part in msg["text"])
        else:
            text = str(msg["text"])

        messages.append({
            "id": msg["id"],
            "date": msg["date"],
            "from": msg.get("from", "Unknown"),
            "text": text.strip()
        })

    return messages


# app/services/analyzer.py

import re
from typing import List, Dict
from textblob import TextBlob
import json

# Загрузка фильтра шумовых сообщений
with open("app/resources/noise_patterns.json", "r", encoding="utf-8") as f:
    NOISE_PATTERNS = [re.compile(p, re.IGNORECASE) for p in json.load(f)["noise_patterns"]]


def is_noise(text: str) -> bool:
    return any(pattern.match(text) for pattern in NOISE_PATTERNS)


def clean_messages(messages: List[Dict]) -> List[Dict]:
    return [msg for msg in messages if not is_noise(msg["text"])]


def analyze_sentiment(text: str) -> Dict[str, float]:
    blob = TextBlob(text)
    return {
        "polarity": blob.sentiment.polarity,
        "subjectivity": blob.sentiment.subjectivity
    }


# app/models/chat.py

from sqlalchemy import Column, Integer, String, Text, DateTime
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class Message(Base):
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, index=True)
    date = Column(DateTime)
    sender = Column(String)
    text = Column(Text)
    polarity = Column(String)        # Для настроения
    subjectivity = Column(String)


# app/db/database.py

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.models.chat import Base

DATABASE_URL = "sqlite:///./chat_data.db"

engine = create_engine(
    DATABASE_URL, connect_args={"check_same_thread": False}
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    Base.metadata.create_all(bind=engine)


"""
<!-- frontend/index.html -->
<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <title>Анализ Telegram-чата</title>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css">
</head>
<body class="bg-light p-4">
    <div class="container">
        <h1 class="mb-4">Импорт Telegram-чата</h1>

        <form action="/upload" method="post" enctype="multipart/form-data" class="mb-3">
            <div class="mb-3">
                <label for="file" class="form-label">Выберите JSON-файл</label>
                <input type="file" class="form-control" id="file" name="file" accept=".json" required>
            </div>
            <button type="submit" class="btn btn-primary">Загрузить</button>
        </form>

        <div id="response" class="mt-4"></div>
    </div>
</body>
</html>

"""