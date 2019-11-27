import shutil
import os
# Каталог с набором данных
data_dir = 'TrashPhoto'
# Каталог с данными для обучения
train_dir = 'train'
# Каталог с данными для проверки
val_dir = 'val'
# Каталог с данными для тестирования
test_dir = 'test'
# Часть набора данных для тестирования
test_data_portion = 0.15
# Часть набора данных для проверки
val_data_portion = 0.15
# Категории мусора
CATEGORIES = ['cardboard', 'glass', 'metal', 'paper', 'plastic']
NUMPHOTOS = [403, 501, 410, 594, 482]

# Создание директорий
def create_directory(dir_name):
    if os.path.exists(dir_name):
        shutil.rmtree(dir_name)
    os.makedirs(dir_name)
    for category in CATEGORIES:
        os.makedirs(os.path.join(dir_name, category))

create_directory(train_dir)
create_directory(val_dir)
create_directory(test_dir)

# Копирование картинок
def copy_images(start_index, end_index, source, dest):
    for i in range(start_index, end_index):   
        shutil.copy2(source + str(i) + ".jpg", dest)

for i in range(len(CATEGORIES)): 
    category = CATEGORIES[i]
    nb_images = NUMPHOTOS[i]
    # Индексы для наборов
    start_val_data_idx = int(nb_images * (1 - val_data_portion - test_data_portion))
    start_test_data_idx = int(nb_images * (1 - test_data_portion))
    # Пути к файлам
    source = os.path.join(data_dir,category, category)

    train_dest = os.path.join(train_dir,category)
    val_dest = os.path.join(val_dir,category)
    test_dest = os.path.join(test_dir,category)

    copy_images(1, start_val_data_idx, source, train_dest)
    copy_images(start_val_data_idx, start_test_data_idx, source, val_dest)
    copy_images(start_test_data_idx, nb_images, source, test_dest)