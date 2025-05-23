# TM-NA

### *создано исключительно в развлекательных целях*
---

# Анализ и визуализация чатов Telegram

Веб-приложение на FastAPI для анализа чатов Telegram. Поддерживает загрузку экспортированного JSON-файла, визуализацию статистики, фильтрацию по авторам и тональностям, построение графиков активности.

---

## Возможности

* Загрузка JSON-файла Telegram-чата
* Генерация уникального ключа для каждой сессии (повторный доступ)
* Фильтрация сообщений по дате, авторам и тональностям
* Визуализация активности по дням, времени суток, дням недели
* Распознавание тональности сообщений (позитивные, нейтральные, негативные)
* Графики: распределение по времени, частотные слова, популярные эмодзи
* Числовая статистика: общее число сообщений, доли разных тональностей, активность по времени
* Персональная статистика по каждому участнику: средняя длина сообщения, пик активности, частота сообщений, индивидуальное облако слов

---

## Установка

1. Клонируйте репозиторий:

```bash
git clone https://github.com/DomiStjls/Telegrasta.git
cd Telegrasta
```

2. Установите зависимости:

```bash
pip install -r requirements.txt
```

3. Запустите сервер:

```bash
uvicorn main:app --reload
```

---

## Структура проекта

* `main.py` — основное FastAPI-приложение
* `templates/` — HTML-шаблоны (Jinja2)
* `static/` — изображения графиков и стили
* `data/{key}/` — временные папки с данными каждой сессии
* `utils/` — вспомогательные функции: генерация графиков, статистика, wordcloud

---

## Использование

1. Экспортируйте чат из Telegram в формате JSON.
2. Перейдите на главную страницу приложения.
3. Загрузите JSON-файл.
4. Получите уникальный ключ сессии.
5. Используйте фильтры и просматривайте статистику.

---

## Инструкция к скачиванию истории чата из Telegram в виде json

1. Нажмите на 3 точки в чате

![image](https://github.com/user-attachments/assets/73b042fe-1e53-4009-a4e9-3ab558c2b958)

2. Нажмите "Экспорт истории чата"

![image](https://github.com/user-attachments/assets/a7376375-0236-4265-9088-44e6a3b0ada1)

3. Выставите нужные параметры (статистика учитывает только текстовые сообщения, поэтому все остальное можно не экспортировать), обязательно выберите формат json

![image](https://github.com/user-attachments/assets/e217ccf2-7d0e-4490-a184-4e2913f43d66)


