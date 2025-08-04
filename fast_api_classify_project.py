import io,httpx
from fastapi import FastAPI, HTTPException
from keras.models import load_model  # TensorFlow is required for Keras to work
from PIL import Image, ImageOps  # Install pillow instead of PIL
import numpy as np
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

class Image1(BaseModel): # BaseModel — базовый класс от pydantic, который: автоматически проверяет типы данных, валидирует входные параметры, удобно сериализует в JSON
                                     # img_url: str — поле, ожидающее строку
    img_url: str

np.set_printoptions(suppress=True) # set_printoptions() — функция, которая меняет поведение print() для numpy массивов
                                   # suppress=True — означает: не показывать числа в экспоненциальной форме

# Загружаем модель
model = load_model("./pneumonia classificatio web/keras_model_xray.h5", compile=False)

# Загружаем метки
class_names = open("./pneumonia classificatio web/labels 1.txt", "r").readlines()

# Загружаем апи и даем ему название
app = FastAPI(title="X-Ray Classification API")
# Важная часть кода без которого апи будет не доступен в разных регионах, делается для того чтобы каждый мог запустить наш апи
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/classify") 
#  Почему именно POST? Потому что ты отправляешь некую информацию (в нашем случае — URL картинки) и ожидаешь результат обработки (например, класс изображения).
# Это не просто "показать страницу", как в GET, а именно вызвать действие.

async def classify(image: Image1): # асинхронная функция где аргумент image — это объект Image1
    image_url = image.img_url      # Достаём URL изображения из нашего запроса.
    
    try:                           # try: Начало блока обработки исключений. Если внутри блока произойдёт ошибка (например, сервер недоступен, неверный URL),
                                            #программа не упадёт, а перейдёт в except
        
        async with httpx.AsyncClient() as cli: # Создаётся асинхронный HTTP-клиент с помощью httpx.AsyncClient()
            resp = await cli.get(image_url, timeout=10) # Отправляется асинхронный GET-запрос по адресу image_url
                                                        # Await говоріт пайтон подожди ответа, но не блокируй всё приложение
                                                        # Если в течении 10 секунд сервер не даст ответа то будет исклчение
        resp.raise_for_status()                         # Проверка: а точно ли сервер вернул «успешный» ответ, если нет то исклчючение
    except Exception as e:                              # А вот и наше исключение переменное в "e" Exception — базовый класс для всех стандартных ошибок Python
                                                        # Сетевые ошибки, ошибки статуса и ошибки преобразования адреса и прочие подобные штуки
        raise HTTPException(400, f"Cannot fetch image: {e}")
        # Мы генерируем исключение HTTP, которое FastAPI автоматически конвертирует в HTTP-ответ.
        # HTTPException — специальный класс FastAPI (из fastapi.exceptions) для возврата контролируемых ошибок клиенту.
        # Аргументы: 400 — HTTP-статус: Bad Request (клиент виноват: передал некорректный URL, например).
        # f"Cannot fetch image: {e}" — сообщение об ошибке, в котором вставляется текст исключения e.

    # 2) Открываем PIL-Image из байтов
    try:
        image = Image.open(io.BytesIO(resp.content)).convert("RGB")  # Пытаемся открыть изображение с помощью PIL
                                                                    # resp.content — это байтовый объект (всё, что сервер вернул по ссылке).
                                     # io.BytesIO() — оборачивает байты в "виртуальный файл", чтобы PIL.Image.open() мог прочитать их как если бы это был файл на диске.
                                     # Image.open() — пытается распознать и открыть изображение.
                                    # конвертируем все в RGB
                                    # если часть кода не корректная выбрасываем исклчение
    except Exception:
        raise HTTPException(status_code=400, detail="URL did not return a valid image") # Делаем такое же исключение как и выше 
                                                                                    # status_code=400 HTTP код 400 Bad Request
                                                                                    # detail= "Сообщение об ошибке (для клиента)"

    # 3) Код предобработки из teachble machine
    size = (224, 224)

    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    image_array = np.asarray(image).astype(np.float32)

    normalized = (image_array / 127.5) - 1

    data = np.ndarray(shape=(1,224,224,3), dtype=np.float32)

    data[0] = normalized

    # 4) Инференс
    prediction = model.predict(data)        # берем нашу модель и делаем предикт по дате
    index      = int(np.argmax(prediction[0])) # np.argmax находит индекс максимального значения в массиве. prediction[0] берем первое значение и int() переводим в число
    label      = class_names[index]            # берем и достаём строковое имя класса у нас это class_names[1] = Normal, class_names[0] = pneumonia 
    score      = float(prediction[0][index])   # берем первое значение из prediction[0] и [index] достаем вероятность предсказаного индекса 
                                               # Приводит к стандартному Python float() Сохраняем результат как обычное число в score

    # 5) Ответ
    return {                                   # возвращаем словарь который FastAPI автоматически подгонит резульатт  
        "prediction": label[2:-1],             # идёт обрезка строки label, начиная с 3-го символа до предпоследнего
        "confidence": round(score,4)           # round(score, 4) округляем резульатт до 4 знаков после запятой, например резлуьатт 0.971857 у нас будет 0.9719
    }

