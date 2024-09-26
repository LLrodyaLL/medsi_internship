import os
import cv2
import albumentations as A

# Пути к папкам
image_dir = r'C:\Users\user\Downloads\new-cabinet510-10m\images'
label_dir = r'C:\Users\user\Downloads\new-cabinet510-10m\labels'
output_image_dir = r'C:\Users\user\Desktop\new-cabinet510-10m_aug\images_aug'
output_label_dir = r'C:\Users\user\Desktop\new-cabinet510-10m_aug\labels_aug'

# Создайте выходные папки, если они не существуют
os.makedirs(output_image_dir, exist_ok=True)
os.makedirs(output_label_dir, exist_ok=True)

# Аугментации
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
    A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
    A.Blur(blur_limit=3, p=0.3),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=0, p=0.5),
    A.Rotate(limit=30, p=0.5),
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))


# Функция для чтения аннотаций в формате YOLO
def read_yolo_annotations(label_path):
    bboxes = []
    class_labels = []
    with open(label_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            class_id, x_center, y_center, width, height = map(float, line.strip().split())
            bboxes.append([x_center, y_center, width, height])
            class_labels.append(int(class_id))
    return bboxes, class_labels


# Функция для записи аннотаций в формате YOLO
def save_yolo_annotations(label_path, bboxes, class_labels):
    with open(label_path, 'w') as file:
        for bbox, class_id in zip(bboxes, class_labels):
            x_center, y_center, width, height = bbox
            file.write(f"{class_id} {x_center} {y_center} {width} {height}\n")


for image_name in os.listdir(image_dir):
    image_path = os.path.join(image_dir, image_name)

    # Предполагаем, что файлы изображений имеют расширения .png или .jpg, а файлы аннотаций — .txt
    label_path = os.path.join(label_dir, image_name.replace('.png', '.txt').replace('.jpg', '.txt'))

    # Проверяем, существует ли файл аннотации
    if not os.path.exists(label_path):
        print(f"Аннотация для {image_name} не найдена, пропускаем этот файл.")
        continue

    # Загружаем изображение и аннотации
    image = cv2.imread(image_path)
    bboxes, class_labels = read_yolo_annotations(label_path)

    # Выполняем аугментацию
    augmented = transform(image=image, bboxes=bboxes, class_labels=class_labels)
    augmented_image = augmented['image']
    augmented_bboxes = augmented['bboxes']
    augmented_class_labels = augmented['class_labels']

    # Сохраняем аугментированное изображение и аннотации
    output_image_path = os.path.join(output_image_dir, 'aug_' + image_name)
    output_label_path = os.path.join(output_label_dir,
                                     'aug_' + image_name.replace('.png', '.txt').replace('.jpg', '.txt'))

    cv2.imwrite(output_image_path, augmented_image)
    save_yolo_annotations(output_label_path, augmented_bboxes, augmented_class_labels)

