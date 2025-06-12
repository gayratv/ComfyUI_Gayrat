# импорт из Comfy

class FloatSlider:

    def __init__(self):
        self.NODE_NAME = 'Float slider'

    @classmethod
    def INPUT_TYPES(self):
        return {"required": {
                    "weight": ("FLOAT", {
                        "default": 1,
                        "min": 0,
                        "max": 1,
                        "step": 0.01,
                        "display": "slider"
                    }),
                },
                "optional": {}
        }
    RETURN_TYPES = ("FLOAT",)
    FUNCTION = "run"
    CATEGORY = "Gayrat"
    DESCRIPTION = "Простой слайдер от 0 до 1"


    def run(self, weight):
        scaled_number = weight
        return (scaled_number,)

NODE_DISPLAY_NAME_MAPPINGS = {
    "FLOATSLIDER": "Float slider"
}

NODE_CLASS_MAPPINGS = {
    "FLOATSLIDER" : FloatSlider,
}