import roboflow

# Указываем API-ключ
rf = roboflow.Roboflow(api_key="lfEZ4ie8OZdMWZhfUXf3")

# Подключаемся к проекту
project = rf.workspace().project("conveyer-belt-vz1tr-riyjb")

# Указываем версию датасета (замени 1 на нужную версию)
dataset_version = 2 

# Скачиваем датасет в указанную папку
dataset_path = r"C:\ConveyorVision\dataset"
dataset = project.version(dataset_version).download(location=dataset_path)

print(f"Датасет загружен в: {dataset_path}")
