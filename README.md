# Design Document: Early Prediction of Spotify Hit Songs

## 1. Goal and Metrics

### Цель

Построить модель, которая по первым дням нахождения песни в Spotify-чартах предсказывает, станет ли песня успешной в ближайшем будущем. Дополнительная исследовательская цель — проверить, улучшают ли музыкальные/audio-признаки качество предсказания.

### Формальная постановка

Объект: пара `track-region` или один трек в Global-чарте.

Целевая переменная:

```text
target_future_top50 = 1,
если трек достиг Top 50 в течение 30 дней после первых 7 дней наблюдения,
иначе 0.
```

Первые 7 дней используются как окно наблюдения. Следующие 30 дней используются только для target.

### Метрики

Основные:

- F1-score;
- ROC-AUC;
- Precision;
- Recall.

Дополнительные:

- Accuracy;
- Average Precision;
- Log Loss;
- Brier Score;
- confusion matrix.

F1 важен, потому что классы могут быть несбалансированы: настоящих будущих хитов меньше, чем обычных треков.

---

## 2. Data

### Источник 1: Spotify Charts

Используются исторические чарты Spotify: Top 200 и/или Global chart.

Ключевые поля:

- `date`;
- `title`;
- `artist`;
- `rank`;
- `region`;
- `chart`;
- `streams`;
- `url`.

### Источник 2: Spotify Tracks Dataset / Audio Features

Ключевые поля:

- `track_id`;
- `track_name`;
- `artists`;
- `danceability`;
- `energy`;
- `valence`;
- `tempo`;
- `loudness`;
- `acousticness`;
- `speechiness`;
- `instrumentalness`;
- `liveness`;
- `duration_ms`;
- `key`;
- `mode`;
- `time_signature`;
- `track_genre`.

### Объединение данных

Chart dataset обычно намного больше, чем audio-features dataset. Поэтому после merge остаются только те треки, которые есть и в чартах, и в audio-features источнике. Чтобы это не ломало проект, используются две модели:

- **Model A**: использует только chart/artist features и может обучаться на большем числе строк;
- **Model B**: использует chart/artist + audio/structure features и обучается только на audio-matched строках.

В таблицу добавлены диагностические колонки:

- `has_audio_features`;
- `audio_features_non_missing`.

Они используются для фильтрации данных, но не используются как признаки модели.

### Ограничения данных

1. В чартах есть только треки, которые уже стали достаточно заметными, чтобы попасть в Top 200.
2. Внешние факторы — маркетинг, TikTok, плейлисты, фан-база — не полностью представлены в данных.
3. Объединение chart data и audio features по названию и артисту может давать ошибки.
4. Spotify audio features не равны настоящей структуре песни, поэтому structure-inspired features являются приближением.
5. Если Model A обучается на всех строках, а Model B только на audio-matched строках, сравнение метрик не полностью честное из-за разного состава данных. Поэтому для главного вывода используется `fair`-режим.

---

## 3. Validation Draft

Нельзя использовать обычный random split, потому что задача связана с будущим.

Используется временная схема валидации:

- сначала данные сортируются по `first_date`;
- самый новый блок данных откладывается как final test set;
- на более старой части данных применяется expanding-window time-series cross-validation;
- в каждом fold модель обучается на прошлом и валидируется на более позднем периоде.

Это имитирует реальный сценарий: обучаемся на прошлых треках и предсказываем будущие. Обычный `KFold` не используется, потому что он перемешивал бы временные периоды и создавал риск data leakage.

---

## 4. Approach

### Главный эксперимент

#### Model A: chart-only

Использует только признаки ранней динамики в чартах и историю артиста. Audio features и structure-inspired features исключаются.

#### Model B: chart + audio

Использует все признаки Model A плюс музыкальные/audio-признаки и простые structure-inspired признаки.

### Режим сравнения

#### Fair mode

Model A и Model B обучаются на одном и том же audio-matched подмножестве. Это честный способ проверить, добавляют ли audio features полезную информацию.

### Baseline

1. `DummyClassifier`: всегда предсказывает самый частый класс.
2. `LogisticRegression` на базовых признаках.

### Основной feature engineering

#### Ранняя динамика в чартах

- first rank;
- best rank during first 7 days;
- mean rank during first 7 days;
- rank change;
- rank trend/slope;
- first streams;
- total streams during first 7 days;
- streams growth;
- streams trend/slope;
- number of observed chart days;
- number of regions where the track appeared during the observation window.

#### История артиста

Для каждого артиста считаются только прошлые события относительно даты первого появления текущего трека:

- previous chart entries;
- previous best rank;
- previous mean streams;
- new artist flag.

#### Музыкальные признаки

- danceability;
- energy;
- valence;
- tempo;
- loudness;
- acousticness;
- speechiness;
- instrumentalness;
- liveness;
- duration;
- key;
- mode;
- time signature;
- genre.

#### Structure-inspired features

Без нейросетей и без полноценного audio segmentation:

- `duration_min`;
- `is_short_track`;
- `is_long_track`;
- `dance_energy`;
- `positive_energy`;
- `loudness_energy`;
- `acoustic_energy_contrast`;
- `tempo_squared`.

---

## 5. Training

Модели:

- DummyClassifier;
- Logistic Regression;
- kNN;
- Decision Tree;
- Random Forest;
- Gradient Boosting.

Preprocessing:

- числовые признаки: imputation + scaling;
- категориальные признаки: imputation + one-hot encoding.

### Feature Selection

# Feature Selection

Отбор признаков реализован как отдельный настраиваемый этап в `src/train.py`.

Он применяется внутри `sklearn Pipeline`:

```text
preprocessing -> feature selection -> final classifier
```

Это важно, потому что отбор признаков обучается только на train-данных во время валидации. Значит, он не использует информацию из validation/test и не создаёт data leakage.

---

## Реализованные режимы

### 1. No formal selection (`none`)

В этом режиме вручную удаляются только:

- metadata;
- identifiers;
- target-like columns.

Это нужно, чтобы предотвратить data leakage.

---

### 2. Filter methods (`filter`)

Сначала `VarianceThreshold` удаляет константные или почти константные признаки после преобразования.

Затем `SelectKBest` оставляет наиболее информативные признаки. По умолчанию используется `f_classif`.

Этот подход быстрый и хорошо подходит для больших датасетов с чартами.

---

### 3. Wrapper methods (`wrapper`)

Используется `RFE` с `LogisticRegression`.

`RFE` многократно обучает модель и постепенно удаляет более слабые признаки.

Этот метод требует больше вычислений, поэтому его лучше использовать:

- после предварительного сокращения пространства признаков;
- на подмножестве треков, которые удалось сопоставить с audio features.

---

### 4. Embedded methods (`embedded`)

Используется `SelectFromModel` с одним из двух вариантов:

- `L1-regularized Logistic Regression`;
- `Random Forest`.

В случае L1-регуляризации слабые коэффициенты могут становиться равными нулю.

В случае Random Forest отбор использует tree-based feature importances.

Это называется embedded-методом, потому что отбор признаков выполняется моделью во время обучения.

---

### 5. Hybrid methods (`hybrid`)

Это подход по умолчанию:

```text
VarianceThreshold -> SelectKBest -> SelectFromModel -> RFE
```

Он объединяет:

- быструю статистическую фильтрацию;
- встроенный модельный отбор;
- рекурсивный обёрточный отбор.

---

## Артефакты

### `feature_selection_report.csv`

Содержит информацию о том, какие признаки были выбраны или отброшены на каждом этапе, включая:

- filter stage;
- embedded stage;
- wrapper stage.

### `selected_transformed_features.json`

Содержит финальный список выбранных признаков после preprocessing и feature selection.

### `strict - версия файлов`

Содержат результаты по предсказанию без использования в датасете песен уже вошедших в топ 50 в первые 7 дней.

### `feature_importance.csv`

Содержит коэффициенты или tree-based importances финальной модели, если выбранная модель это поддерживает.
Подбор гиперпараметров можно добавить на следующем этапе через `GridSearchCV`, но для временных данных нужно использовать time-series folds, а не random KFold.


---
## 6. Validation

Для каждой модели считаются метрики на нескольких временных CV-folds. Лучшая модель выбирается по среднему значению выбранной метрики, например `f1_mean` или `roc_auc_mean`.

После выбора лучшая модель переобучается на всей старой части данных, то есть `train + validation`, и оценивается на final test set, который не участвовал ни в обучении, ни в выборе модели.

Артефакты кросс-валидации:

- `metrics_cv_folds.csv`: метрики каждой модели на каждом fold;
- `metrics_cv_summary.csv`: среднее и стандартное отклонение метрик по fold;
- `metrics_test.csv`: финальная оценка выбранной модели на отложенном test set.

---

## 7. Model Selection

Финальная модель выбирается не только по accuracy, а по F1 и ROC-AUC.

Основной исследовательский вывод строится по таблице:

```text
reports/model_comparison_fair.csv

reports_strict/model_comparison_fair.csv
```
Сравнение моделей можно посмотреть в таблицах:
```text
reports\model_B_chart_plus_audio_matched\metrics_validation.csv

reports_strict\model_B_chart_only_matched\metrics_validation.csv

или с буквой A
```

Для варианта A наилучшей моделью стала - gradient_boosting, для B - random_forest.
Модель B показала чуть лучшие результаты по основным метрикам, но разница настолько незначительна, что можно сделать вывод о малой полезности признаков, отражающих структуру трека. Разве, что после отбора признаков для варианта B остались те категориальные признаки, которые отражают жанр композиции (1 - если это тот жанр, 0 - если не тот). 

Ожидаемые риски:

1. **Selection bias**: датасет содержит только песни, попавшие в чарты.
2. **Data leakage**: нельзя использовать данные после окна наблюдения.
3. **Artist popularity bias**: модель может слишком сильно полагаться на историю артиста.
4. **Audio matching bias**: audio features доступны только для части треков.

---

## 8. Deploy

Возможный интерфейс:

- Streamlit dashboard;
- выбор папки модели A или B;
- загрузка CSV с признаками;
- вывод вероятности будущего успеха для каждого трека.


