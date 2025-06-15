import asyncio
from googletrans import Translator


# Функция для выполнения блокирующего кода перевода.
# Мы выносим ее отдельно, чтобы было удобнее передать в executor.
def do_translation(text):
    if not text.strip():
        return ""
    try:
        translator = Translator()
        translation = translator.translate(text, dest='en')
        return translation.text
    except Exception as e:
        print(f"Ошибка перевода: {e}")
        return f"Ошибка: {e}"


class GoogleTranslateNode2:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text_1": ("STRING", {"multiline": True, "default": "Привет, мир!"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("translated_text",)

    # Меняем имя функции на асинхронное
    FUNCTION = "translate_text_async"

    CATEGORY = "Text"

    # Создаем асинхронную версию нашей функции, добавляя 'async def'
    async def translate_text_async(self, text_1):
        # Получаем текущий цикл событий asyncio, на котором работает ComfyUI
        loop = asyncio.get_running_loop()

        # Запускаем нашу блокирующую функцию 'do_translation' в отдельном потоке
        # и ждем ее завершения, не блокируя основной поток.
        # 'None' в качестве первого аргумента означает использование executor'а по умолчанию (ThreadPoolExecutor).
        translated_text = await loop.run_in_executor(
            None, do_translation, text_1
        )

        # Возвращаем результат точно так же, как и раньше
        return (translated_text, {"ui": {"translated_text": [translated_text]}})


# --- Регистрация узла остается без изменений ---
NODE_CLASS_MAPPINGS = {
    "GoogleTranslateNode2": GoogleTranslateNode2
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "GoogleTranslateNode2": "Google Translate2"
}