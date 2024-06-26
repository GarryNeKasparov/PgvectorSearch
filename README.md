# PgvectorSearch
Поисковой движок для PostgreSQL на основе расширения pgvector.

![image](https://github.com/GarryNeKasparov/PgvectorSearch/assets/52318064/435d7518-6b5c-4eb0-8ae7-76f61cd74ab2)

## Структура

* [scoring.py](db_project/scoring.py) - оценка качества представлений.
* [utils.py](db_project/utils.py) - вспомогательные функции для работы с базой данных.

`word2vec`:
* [main.py](db_project/word2vec/main.py) - получение представлений.
* [query.py](db_project/word2vec/query.py) - интерактивный поиск товаров. 

`lstm`:
* [main.py](db_project/lstm/main.py) - получение представлений.
* [data.py](db_project/lstm/data.py) - подготовка данных для обучения модели.
* [train.py](db_project/lstm/train.py) - обучение модели.
* [models.py](db_project/lstm/models.py) - реализация архитектуры модели LSTM.
* [query.py](db_project/lstm/query.py) - интерактивный поиск товаров.

`transf`:
* [main.py](db_project/transf/main.py) - получение представлений моделью MiniLM.

`sql`:
* [listening.py](db_project/sql/listening.py) - реализация механизма обработки новых товаров через отслеживание уведомлений.
* скрипты sql для постоения индекса, триггеров и т.д.

`src`:
* некоторые графики оценки эффективности моделей
