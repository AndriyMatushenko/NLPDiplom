# -*- coding: utf-8 -*-
"""
Скрипт для профілювання продуктивності додатку "Юридичний асистент".

Дозволяє тестувати окремо процес первинної обробки даних
та процес обробки одного користувацького запиту.

Використання:
1. Профілювання обробки даних:
   python -m cProfile -s tottime profile_app.py --test-processing

2. Профілювання обробки запиту:
   python -m cProfile -s tottime profile_app.py --test-query
"""
import argparse
import os

# Важливо імпортувати функції з вашого основного додатку
from app import process_data, Search, generate_answer

# Видаляємо старий файл з даними для чистоти експерименту
if os.path.exists('documents.pkl'):
    os.remove('documents.pkl')


def run_processing_test():
    """Тестує функцію первинної обробки даних."""
    print("--- ЗАПУСК СЦЕНАРІЮ A: ПРОФІЛЮВАННЯ ПЕРВИННОЇ ОБРОБКИ ДАНИХ ---")
    process_data()
    print("--- СЦЕНАРІЙ A ЗАВЕРШЕНО ---")


def run_query_test():
    """Тестує процес обробки одного запиту."""
    print("--- ЗАПУСК СЦЕНАРІЮ B: ПРОФІЛЮВАННЯ ОБРОБКИ ОДНОГО ЗАПИТУ ---")

    # Переконуємось, що дані існують
    if not os.path.exists('documents.pkl'):
        process_data()

    search_engine = Search()
    test_query = "Яке покарання передбачене за крадіжку?"

    print(f"Тестовий запит: '{test_query}'")

    context = search_engine.query(test_query)
    answer = generate_answer(context, test_query)

    print(f"Отримана відповідь: {answer[:100]}...")
    print("--- СЦЕНАРІЙ B ЗАВЕРШЕНО ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Скрипт для профілювання додатку 'Юридичний асистент'."
    )
    parser.add_argument(
        '--test-processing',
        action='store_true',
        help='Запустити профілювання первинної обробки даних.'
    )
    parser.add_argument(
        '--test-query',
        action='store_true',
        help='Запустити профілювання обробки одного запиту.'
    )

    args = parser.parse_args()

    if args.test_processing:
        run_processing_test()
    elif args.test_query:
        run_query_test()
    else:
        print("Будь ласка, вкажіть, що тестувати: --test-processing або --test-query")
