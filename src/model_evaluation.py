import numpy as np
from keras.models import load_model
from keras.preprocessing import image
from sklearn.metrics import classification_report, confusion_matrix

def evaluate_model(model_path, val_dir):
    model = load_model(model_path)
    
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
    validation_generator = datagen.flow_from_directory(
        val_dir,
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary',
        subset='validation'
    )
    
    validation_generator.reset()
    Y_pred = model.predict(validation_generator)
    y_pred = np.where(Y_pred > 0.5, 1, 0)
    
    print('Classification Report')
    target_names = ['Normal', 'Pneumonia']
    print(classification_report(validation_generator.classes, y_pred, target_names=target_names))
    
    print('Confusion Matrix')
    print(confusion_matrix(validation_generator.classes, y_pred))

if __name__ == "__main__":
    model_path = 'models/model_pneumonia_detection.h5'
    val_dir = 'data/val'
    
    evaluate_model(model_path, val_dir)

