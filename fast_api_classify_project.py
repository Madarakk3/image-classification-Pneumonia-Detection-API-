import io,httpx
from fastapi import FastAPI, HTTPException
from keras.models import load_model  # TensorFlow is required for Keras to work
from PIL import Image, ImageOps  # Install pillow instead of PIL
import numpy as np
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

class Image1(BaseModel):
    img_url: str

np.set_printoptions(suppress=True)

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


@app.post("/classify")  # сделали GET
async def classify(image: Image1):
    image_url = image.img_url
    try:
        async with httpx.AsyncClient() as cli:
            resp = await cli.get(image_url, timeout=10)
        resp.raise_for_status()
    except Exception as e:
        raise HTTPException(400, f"Cannot fetch image: {e}")

    # 2) Открываем PIL-Image из байтов
    try:
        image = Image.open(io.BytesIO(resp.content)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="URL did not return a valid image")

    # 3) Код предобработки из teachble machine
    size = (224, 224)

    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    image_array = np.asarray(image).astype(np.float32)

    normalized = (image_array / 127.5) - 1

    data = np.ndarray(shape=(1,224,224,3), dtype=np.float32)

    data[0] = normalized

    # 4) Инференс
    prediction = model.predict(data)
    index      = int(np.argmax(prediction[0]))
    label      = class_names[index]
    score      = float(prediction[0][index])

    # 5) Ответ
    return {
        "prediction": label[2:-1],
        "confidence": round(score,4)
    }

