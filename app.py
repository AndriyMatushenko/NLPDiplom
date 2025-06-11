# -*- coding: utf-8 -*-
"""
Юридичний асистент на основі RAG.

Цей застосунок збирає юридичні дані з українських сайтів, обробляє їх
та використовує локальну LLM для надання відповідей на запити користувачів.
"""
import logging
import logging.handlers
import os
import pickle
import re
import ssl
import threading
import uuid

import gradio as gr
import nltk
import numpy as np
import pymorphy3
import requests
import sentry_sdk  # Додано Sentry SDK
from bs4 import BeautifulSoup
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer

import ollama

# --- Ініціалізація Sentry ---
SENTRY_DSN = os.getenv('SENTRY_DSN')
if SENTRY_DSN:
    sentry_sdk.init(
        dsn=SENTRY_DSN,
        # Вмикає збір додаткової інформації про продуктивність
        traces_sample_rate=1.0,
        profiles_sample_rate=1.0,
    )
    print("Sentry SDK ініціалізовано.")
else:
    print("SENTRY_DSN не знайдено. Sentry не буде використовуватись.")

# --- Розширене налаштування логування ---
local_storage = threading.local()


class RequestIdFormatter(logging.Formatter):
    """Кастомний форматер, що додає унікальний ID запиту до кожного лог-запису."""

    def format(self, record):
        record.request_id = getattr(local_storage, 'request_id', 'main-thread')
        return super().format(record)


def setup_logging():
    """Налаштовує систему логування з двома обробниками та кастомним форматом."""
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO').upper()
    root_logger = logging.getLogger()
    root_logger.setLevel(LOG_LEVEL)

    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    formatter = RequestIdFormatter(
        '%(asctime)s - %(request_id)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    file_handler = logging.handlers.RotatingFileHandler(
        'app.log', maxBytes=5 * 1024 * 1024, backupCount=5, encoding='utf-8'
    )
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)


setup_logging()
logger = logging.getLogger(__name__)

# --- Кінець налаштування ---


# Налаштування SSL для NLTK
# pylint: disable=protected-access
try:
    _create_unverified_https_context = ssl._create_unverified_context
    ssl._create_default_https_context = _create_unverified_https_context
except AttributeError:
    pass

# Константи
DOCUMENTS_FILE = 'documents.pkl'
CHUNK_SIZE = 7
TOP_K = 5
SEARCH_SOURCES = [
    'https://zakon.rada.gov.ua/laws/show/2341-14#Text',
    'https://zakon.rada.gov.ua/laws/show/80731-10#Text',
    'https://zakon.rada.gov.ua/laws/show/80732-10#Text',
]

USER_AGENT = (
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
    '(KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
)
REQUEST_HEADERS = {'User-Agent': USER_AGENT}

MORPH = pymorphy3.MorphAnalyzer()


def scrape_legal_data():
    """Завантажує та очищує тексти юридичних документів."""
    documents = []
    logger.info("Початок збору даних з %d джерел.", len(SEARCH_SOURCES))
    for url in SEARCH_SOURCES:
        try:
            logger.debug("Завантаження даних з URL: %s", url)
            resp = requests.get(url, headers=REQUEST_HEADERS, timeout=10)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, 'html.parser')
            content = soup.get_text()
            content = re.sub(r'\s+', ' ', content)
            documents.append({'source': url, 'text': content})
            logger.info("Успішно завантажено та оброблено: %s", url)
        except requests.exceptions.RequestException as e:
            error_id = uuid.uuid4()
            logger.error(
                "Помилка при зборі даних. ID помилки: %s. URL: %s.",
                error_id, url, exc_info=True
            )
            sentry_sdk.capture_exception(e)  # Відправляємо помилку в Sentry
    logger.info("Збір даних завершено. Отримано %d документів.", len(documents))
    return documents


def download_nltk_resources():
    """Перевіряє наявність ресурсів NLTK (punkt) та завантажує їх за потреби."""
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        logger.info("Завантаження ресурсів NLTK 'punkt'...")
        nltk.download('punkt', quiet=True)
        logger.info("Ресурси NLTK 'punkt' завантажено.")


def split_chunks(documents):
    """Розбиває довгі тексти документів на менші сегменти (чанки)."""
    download_nltk_resources()
    chunks = []
    for doc in documents:
        sentences = nltk.sent_tokenize(doc['text'], language='ukrainian')
        for i in range(0, len(sentences), CHUNK_SIZE):
            chunk = ' '.join(sentences[i:i + CHUNK_SIZE])
            chunks.append({'source': doc['source'], 'chunk_text': chunk})
    logger.debug("Розбито на %d чанків.", len(chunks))
    return chunks


def lemmatize(text):
    """Проводить лематизацію тексту."""
    download_nltk_resources()
    words = nltk.word_tokenize(text.lower())
    return [MORPH.parse(w)[0].normal_form for w in words if w.isalpha()]


def process_data():
    """Виконує повний цикл обробки даних."""
    logger.info("Початок процесу обробки даних...")
    docs = scrape_legal_data()

    if not docs:
        error_id = uuid.uuid4()
        logger.critical(
            "Не вдалося зібрати жодного документа. Робота програми неможлива. ID помилки: %s",
            error_id
        )
        raise RuntimeError(
            f"Критична помилка: не вдалося зібрати документи. ID помилки: {error_id}"
        )

    logger.info("Початок розбиття на сегменти...")
    chunks = split_chunks(docs)
    logger.info("Початок лематизації...")
    lemmatized_bm25 = [lemmatize(chunk['chunk_text']) for chunk in chunks]
    lemmatized_tfidf = [' '.join(toks) for toks in lemmatized_bm25]

    logger.info("Збереження оброблених даних у файл %s...", DOCUMENTS_FILE)
    with open(DOCUMENTS_FILE, 'wb') as f:
        pickle.dump({
            'original_chunks': chunks,
            'lemmatized_bm25': lemmatized_bm25,
            'lemmatized_tfidf': lemmatized_tfidf,
        }, f)
    logger.info("Дані успішно оброблено та збережено!")


# pylint: disable=too-few-public-methods
class Search:
    """Клас для індексації даних та виконання пошуку."""

    def __init__(self):
        """Ініціалізує пошуковий рушій."""
        logger.info("Ініціалізація пошукового рушія...")
        try:
            with open(DOCUMENTS_FILE, 'rb') as f:
                data = pickle.load(f)
            logger.info("Дані успішно завантажено з %s.", DOCUMENTS_FILE)
        except (FileNotFoundError, pickle.PickleError, EOFError) as e:
            error_id = uuid.uuid4()
            logger.warning(
                "Не вдалося прочитати файл %s. ID помилки: %s. Деталі: %s. Запускаю повторну обробку.",
                DOCUMENTS_FILE, error_id, e
            )
            process_data()
            with open(DOCUMENTS_FILE, 'rb') as f:
                data = pickle.load(f)

        self.original_chunks = data['original_chunks']
        self.lemmatized_bm25 = data['lemmatized_bm25']

        self.bm25 = BM25Okapi(self.lemmatized_bm25)
        self.tfidf = TfidfVectorizer()
        self.tfidf_matrix = self.tfidf.fit_transform(data['lemmatized_tfidf'])
        logger.info("Пошуковий рушій успішно ініціалізовано.")

    def query(self, text):
        """Виконує пошук найбільш релевантних чанків."""
        logger.debug("Виконання пошукового запиту: '%s'", text)
        lemmas = lemmatize(text)
        bm25_scores = self.bm25.get_scores(lemmas)
        num_available_chunks = len(self.original_chunks)
        num_to_retrieve = min(TOP_K, num_available_chunks)

        if num_to_retrieve == 0:
            logger.warning("Пошук не дав результатів, оскільки в базі немає чанків.")
            return "На жаль, у базі немає достатньої інформації."

        top_ids = np.argsort(bm25_scores)[::-1][:num_to_retrieve]
        results = [self.original_chunks[idx]['chunk_text'] for idx in top_ids]
        logger.debug("Знайдено %d релевантних чанків.", len(results))
        return '\n---\n'.join(results)


def generate_answer(context, query_text):
    """Генерує відповідь на запит, використовуючи LLM."""
    logger.debug("Генерація відповіді для запиту: '%s'", query_text)
    prompt = f"""
Контекст:
{context}

Запит:
{query_text}

Дай точну відповідь, спираючись тільки на інформацію з контексту.
"""
    try:
        res = ollama.chat(
            model="mistral",
            messages=[{"role": "user", "content": prompt}]
        )
        logger.info("Відповідь успішно згенеровано.")
        return res['message']['content']
    except Exception as e:
        error_id = uuid.uuid4()
        # Встановлюємо контекст для звіту Sentry
        with sentry_sdk.push_scope() as scope:
            scope.set_tag("request_id", getattr(local_storage, 'request_id', 'N/A'))
            scope.set_extra("query_text", query_text)
            sentry_sdk.capture_exception(e)

        logger.error(
            "Невідома помилка при генерації відповіді. ID: %s. Запит: '%s'.",
            error_id, query_text, exc_info=True
        )
        return (f"**Вибачте, сталася непередбачувана помилка.**\n\n"
                f"Наша команда вже отримала сповіщення та працює над її вирішенням. "
                f"Будь ласка, спробуйте ще раз пізніше.\n\n"
                f"*Для звернення до підтримки, ви можете вказати ID помилки: `{error_id}`*")


def main():
    """Головна функція програми."""
    logger.info("Запуск застосунку 'Юридичний асистент'.")

    try:
        if not os.path.exists(DOCUMENTS_FILE):
            logger.warning("Файл з даними %s не знайдено. Запуск первинної обробки.", DOCUMENTS_FILE)
            process_data()

        search_engine = Search()

        def main_interface(query_text):
            """Внутрішня функція-обгортка для інтерфейсу Gradio."""
            request_id = f"req-{uuid.uuid4().hex[:8]}"
            local_storage.request_id = request_id

            # Встановлюємо тег request_id для всіх подій Sentry у цьому запиті
            sentry_sdk.set_tag("request_id", request_id)

            logger.info("Отримано новий запит від користувача: '%s'", query_text)
            context = search_engine.query(query_text)
            answer = generate_answer(context, query_text)

            # Очищуємо сховище та теги після завершення запиту
            if hasattr(local_storage, 'request_id'):
                del local_storage.request_id
            sentry_sdk.set_tag("request_id", None)
            return answer

        iface = gr.Interface(
            fn=main_interface,
            inputs=gr.Textbox(
                lines=4,
                placeholder="Наприклад: 'Які права має споживач при поверненні товару?'",
                label="Ваш запит"
            ),
            outputs=gr.Markdown(label="Відповідь юридичного асистента"),
            title="⚖️ Юридичний асистент",
            description="Інформаційна система автоматизованого збору юридичних даних."
        )

        logger.info("Запуск веб-інтерфейсу Gradio...")
        iface.launch()

    except Exception as e:
        error_id = uuid.uuid4()
        logger.critical(
            "Критична неперехоплена помилка під час запуску програми. ID: %s",
            error_id, exc_info=True
        )
        sentry_sdk.capture_exception(e)  # Відправляємо критичну помилку в Sentry
    finally:
        logger.info("Застосунок 'Юридичний асистент' завершив роботу.")


if __name__ == '__main__':
    main()
