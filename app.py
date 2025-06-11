# -*- coding: utf-8 -*-
import os
import re
import ssl
import sys
import nltk
import pickle
import requests
import pymorphy3
import numpy as np
import gradio as gr

from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi
import ollama

# Налаштування SSL для NLTK
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
    'https://zakon.rada.gov.ua/laws/show/2341-14#Text',  # Кримінальний кодекс України
    'https://zakon.rada.gov.ua/laws/show/80731-10#Text',  # Кодекс України про адміністративні правопорушення(частина 1)
    'https://zakon.rada.gov.ua/laws/show/80732-10#Text',  # Кодекс України про адміністративні правопорушення(частина 2)
]

REQUEST_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

morph = pymorphy3.MorphAnalyzer()


# Завантаження юридичних документів
def scrape_legal_data():
    documents = []
    for url in SEARCH_SOURCES:
        try:
            resp = requests.get(url, headers=REQUEST_HEADERS)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, 'html.parser')
            content = soup.get_text()
            content = re.sub(r'\s+', ' ', content)
            documents.append({'source': url, 'text': content})
            print(f"✅ Успішно завантажено: {url}")
        except Exception as e:
            print(f"❌ Помилка при зборі з {url}: {e}")
    return documents


# Завантаження ресурсів NLTK
def download_nltk_resources():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)


# Розбиття тексту на сегменти
def split_chunks(documents):
    download_nltk_resources()
    chunks = []
    for doc in documents:
        sentences = nltk.sent_tokenize(doc['text'], language='ukrainian')
        for i in range(0, len(sentences), CHUNK_SIZE):
            chunk = ' '.join(sentences[i:i + CHUNK_SIZE])
            chunks.append({'source': doc['source'], 'chunk_text': chunk})
    return chunks


# Лематизація тексту
def lemmatize(text):
    download_nltk_resources()
    words = nltk.word_tokenize(text.lower())
    return [morph.parse(w)[0].normal_form for w in words if w.isalpha()]


# Обробка даних
def process_data():
    print("\n▶ Збір юридичних даних...")
    docs = scrape_legal_data()

    if not docs:
        raise RuntimeError("Не вдалося зібрати документи. Перевірте підключення до мережі або джерела.")

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


# Клас для пошуку
class Search:
    def __init__(self):
        try:
            with open(DOCUMENTS_FILE, 'rb') as f:
                data = pickle.load(f)
        except Exception as e:
            print(f"⚠️ Помилка читання файлу: {e}. Повторна обробка даних...")
            process_data()
            with open(DOCUMENTS_FILE, 'rb') as f:
                data = pickle.load(f)

        self.original_chunks = data['original_chunks']
        self.lemmatized_bm25 = data['lemmatized_bm25']
        self.lemmatized_tfidf = data['lemmatized_tfidf']

        self.bm25 = BM25Okapi(self.lemmatized_bm25)
        self.tfidf = TfidfVectorizer()
        self.tfidf_matrix = self.tfidf.fit_transform(self.lemmatized_tfidf)

    def query(self, text):
        lemmas = lemmatize(text)
        bm25_scores = self.bm25.get_scores(lemmas)
        num_available_chunks = len(self.original_chunks)
        num_to_retrieve = min(TOP_K, num_available_chunks)

        if num_to_retrieve == 0:
            return "На жаль, у базі немає достатньої інформації."

        top_ids = np.argsort(bm25_scores)[::-1][:num_to_retrieve]
        results = []
        for idx in top_ids:
            results.append(self.original_chunks[idx]['chunk_text'])
        return '\n---\n'.join(results)


# Генерація відповіді
def generate_answer(context, query):
    prompt = f"""
Контекст:
{context}

Запит:
{query}

Дай точну відповідь, спираючись тільки на інформацію з контексту.
"""
    try:
        res = ollama.chat(model="mistral", messages=[{"role": "user", "content": prompt}])
        return res['message']['content']
    except Exception as e:
        return f"❌ Помилка при генерації відповіді: {e}"


# Головний блок запуску програми
if __name__ == '__main__':
    if not os.path.exists(DOCUMENTS_FILE):
        process_data()

    print("\n▶ Ініціалізація пошукового модуля...")
    search_engine = Search()
    print("✅ Готово!")


    def main_interface(query):
        context = search_engine.query(query)
        answer = generate_answer(context, query)
        return answer


iface = gr.Interface(
    fn=main_interface,
    inputs=gr.Textbox(lines=4, placeholder="Наприклад: 'Які права має споживач при поверненні товару?'", label="Ваш запит"),
    outputs=gr.Markdown(label="Відповідь юридичного асистента"),
    title="⚖️ Юридичний асистент",
    description="Інформаційна система автоматизованого збору україномовних юридичних даних. Введіть запит для отримання юридичної відповіді на основі законів України."
)

print("\n▶ Запуск веб-інтерфейсу... Відкрийте посилання у браузері.")
iface.launch()
