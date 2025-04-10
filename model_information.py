class ModelInfo:
    def __init__(self):
        self.info = {
            "Model name": "DetectionModel",
            "Number of classes": 2,
            "Class names": {0: "Bad", 1: "Good"},
            "Number of parameters": 3006038,
            "Inference time (ms)": 118.26086044311523,
            "Описание": "Модель обучена и настроена для детекции повреждений конвейерной ленты.",
            "Описание классов": {
                0: "Bad - объект, который классифицируется как 'плохой' (например, повреждение на конвейерной ленте)",
                1: "Good - объект, который классифицируется как 'хороший' (например, участок конвейерной ленты без повреждений)"
            }
        }

    def display_model_info(self):
        for key, value in self.info.items():
            print(f"{key}: {value}")

# Пример использования
if __name__ == "__main__":
    model_info = ModelInfo()
    model_info.display_model_info()
