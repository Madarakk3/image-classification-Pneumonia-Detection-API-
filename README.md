# Pneumonia Detection API 🩺

Цей проєкт — це FastAPI-сервер для автоматичної класифікації рентген-знімків грудної клітки з метою виявлення пневмонії. 
Сервер приймає зображення та повертає результат класифікації на основі попередньо натренованої моделі Keras.

---

## 📂 Структура

- `fast_api_classify_project.py` — FastAPI застосунок
- `keras_model_xray.h5` — збережена модель для класифікації
- `labels 1.txt` — текстовий файл з класами (наприклад: NORMAL, PNEUMONIA)
- `example.png` — приклад роботи API
- `chest_xray/` — структура зображень (необов’язкова для запуску API)
- `requirements.txt` — залежності для розгортання

---

## 🚀 Як запустити

```bash
pip install -r requirements.txt
uvicorn fast_api_classify_project:app --reload
```

---

## 📬 Використання API

### POST `/classify`

Приймає зображення грудної клітки (рентген), класифікує його як `NORMAL` або `PNEUMONIA`.

#### 📤 Приклад запиту (curl):
```bash
curl -X POST http://127.0.0.1:8000/classify/ \
  -F "file=@example.png"
```

#### 📥 Відповідь:
```json
{
  "class_name": "PNEUMONIA",
  "confidence": 0.987
}
```

---

## 🧠 Модель

- Навчена на датасеті `chest_xray` (Kaggle / NIH ChestX-ray14)
- Архітектура: TensorFlow + Keras
- Вивід: клас з найбільшою ймовірністю

---

## 📌 Примітка

- API розроблений для тестування моделі в умовах продакшену
- Підходить як backend-сервіс для медичних веб-інтерфейсів або аналітики

---

## 🖼️ Приклад вхідного зображення

![preview](example.png)

---

## 📄 Ліцензія

MIT


---

## 🧰 Створення моделі

Модель була створена за допомогою платформи [Teachable Machine від Google](https://teachablemachine.withgoogle.com), 
що дозволяє швидко створювати нейромережі без глибоких знань у ML.

Для навчання було використано датасет `chest_xray`, доступний, наприклад, на [Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) 
або в NIH ChestX-ray14.

🛠 Якщо бажаєте, ви можете **створити власну модель** на сайті Teachable Machine, експортувати її у формат Keras (`.h5`) 
та замінити файл `keras_model_xray.h5`.

Якщо вам не потрібно навчати свою — готова модель вже в репозиторії ✅.