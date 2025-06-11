# -*- coding: utf-8 -*-
"""
Юридичний асистент на основі RAG.

Цей застосунок збирає юридичні дані з українських сайтів, обробляє їх
та використовує локальну LLM для надання відповідей на запити користувачів.
"""
import os
import pickle
import re
import ssl

import gradio as gr
import nltk
import numpy as np
import pymorphy3
import requests
from bs4 import BeautifulSoup
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer

import ollama

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
    """
    Завантажує тексти юридичних документів із визначених джерел.
    """
    documents = []
    for url in SEARCH_SOURCES:
        try:
            resp = requests.get(url, headers=REQUEST_HEADERS, timeout=10)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, 'html.parser')
            content = soup.get_text()
            content = re.sub(r'\s+', ' ', content)
            documents.append({'source': url, 'text': content})
            print(f"✅ Успішно завантажено: {url}")
        except requests.exceptions.RequestException as e:
            print(f"❌ Помилка при зборі з {url}: {e}")
    return documents


def download_nltk_resources():
    """
    Перевіряє наявність ресурсів NLTK (punkt) та завантажує їх за потреби.
    """
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)


def split_chunks(documents):
    """
    Розбиває довгі тексти документів на менші сегменти (чанки).
    """
    download_nltk_resources()
    chunks = []
    for doc in documents:
        sentences = nltk.sent_tokenize(doc['text'], language='ukrainian')
        for i in range(0, len(sentences), CHUNK_SIZE):
            chunk = ' '.join(sentences[i:i + CHUNK_SIZE])
            chunks.append({'source': doc['source'], 'chunk_text': chunk})
    return chunks


def lemmatize(text):
    """
    Проводить лематизацію тексту: приводить слова до їхньої базової форми.
    """
    download_nltk_resources()
    words = nltk.word_tokenize(text.lower())
    return [MORPH.parse(w)[0].normal_form for w in words if w.isalpha()]


def process_data():
    """
    Виконує повний цикл обробки даних: збір, сегментація, лематизація та збереження.
    """
    print("\n▶ Збір юридичних даних...")
    docs = scrape_legal_data()

    if not docs:
        raise RuntimeError(
            "Не вдалося зібрати документи. Перевірте підключення до мережі."
        )

    print("\n▶ Розбиття на сегменти...")
    chunks = split_chunks(docs)
    print("\n▶ Лемматизація...")
    lemmatized_bm25 = [lemmatize(chunk['chunk_text']) for chunk in chunks]
    lemmatized_tfidf = [' '.join(toks) for toks in lemmatized_bm25]
    with open(DOCUMENTS_FILE, 'wb') as f:
        pickle.dump({
            'original_chunks': chunks,
            'lemmatized_bm25': lemmatized_bm25,
            'lemmatized_tfidf': lemmatized_tfidf,
        }, f)
    print("\n✅ Дані успішно оброблено та збережено!")


# pylint: disable=too-few-public-methods
class Search:
    """
    Клас для індексації даних та виконання пошуку за текстовими запитами.
    """
    def __init__(self):
        """Ініціалізує пошуковий рушій, завантажуючи або створюючи дані."""
        try:
            with open(DOCUMENTS_FILE, 'rb') as f:
                data = pickle.load(f)
        except (FileNotFoundError, pickle.PickleError, EOFError) as e:
            print(f"⚠️ Помилка читання файлу: {e}. Повторна обробка даних...")
            process_data()
            with open(DOCUMENTS_FILE, 'rb') as f:
                data = pickle.load(f)

        self.original_chunks = data['original_chunks']
        self.lemmatized_bm25 = data['lemmatized_bm25']

        self.bm25 = BM25Okapi(self.lemmatized_bm25)
        self.tfidf = TfidfVectorizer()
        self.tfidf_matrix = self.tfidf.fit_transform(data['lemmatized_tfidf'])

    def query(self, text):
        """
        Виконує пошук найбільш релевантних чанків за текстовим запитом.
        """
        lemmas = lemmatize(text)
        bm25_scores = self.bm25.get_scores(lemmas)
        num_available_chunks = len(self.original_chunks)
        num_to_retrieve = min(TOP_K, num_available_chunks)

        if num_to_retrieve == 0:
            return "На жаль, у базі немає достатньої інформації."

        top_ids = np.argsort(bm25_scores)[::-1][:num_to_retrieve]
        results = [self.original_chunks[idx]['chunk_text'] for idx in top_ids]
        return '\n---\n'.join(results)


def generate_answer(context, query_text):
    """
    Генерує відповідь на запит, використовуючи наданий контекст та LLM.
    """
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
        return res['message']['content']
    except ollama.ResponseError as e:
        return f"❌ Помилка API Ollama: {e.error}"
    except Exception as e:
        return f"❌ Невідома помилка при генерації відповіді: {e}"


def main():
    """
    Головна функція програми, що ініціалізує компоненти та запускає інтерфейс.
    """
    if not os.path.exists(DOCUMENTS_FILE):
        process_data()

    print("\n▶ Ініціалізація пошукового модуля...")
    search_engine = Search()
    print("✅ Готово!")

    def main_interface(query_text):
        """
        Внутрішня функція для інтерфейсу Gradio.
        """
        context = search_engine.query(query_text)
        answer = generate_answer(context, query_text)
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
        description=(
            "Інформаційна система автоматизованого збору україномовних "
            "юридичних даних. Введіть запит для отримання юридичної "
            "відповіді на основі законів України."
        )
    )

    print("\n▶ Запуск веб-інтерфейсу... Відкрийте посилання у браузері.")
    iface.launch()


if __name__ == '__main__':
    main()
