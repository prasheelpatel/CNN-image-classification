# Convolutional Neural Network (CNN) models for image classification

CIFAR-10 dataset

Model 1: 
+ Test Accuraccy= 68%
```python
model_1 = keras.models.Sequential([
    keras.layers.Conv2D(32, kernel_size=5, activation='relu', strides = (2,2), padding='same',input_shape=x_train.shape[1:]), 
    keras.layers.Conv2D(32, kernel_size=5, activation='relu', strides = (2,2), ), 
    keras.layers.MaxPooling2D(pool_size=(2, 2)), 
    keras.layers.Dropout(0.25),
    keras.layers.Flatten(),  
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(num_classes, activation='softmax'), 
])
```

Model 2:
+ Batch normalization, Kernel regularization and learning_rate Scheduler
+ Test Accuraccy= 88%
```python
wdecay=1e-4
model_2 = keras.models.Sequential([
    keras.layers.Conv2D(32, kernel_size=3,padding='same', activation='relu',kernel_regularizer=keras.regularizers.l2(wdecay), input_shape=x_train.shape[1:]), 
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(32, kernel_size=3,padding='same', activation='relu',kernel_regularizer=keras.regularizers.l2(wdecay)), 
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D(pool_size=(2, 2)), 
    keras.layers.Dropout(0.2),

    keras.layers.Conv2D(64, kernel_size=3,padding='same', activation='relu',kernel_regularizer=keras.regularizers.l2(wdecay)), 
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(64, kernel_size=3,padding='same', activation='relu',kernel_regularizer=keras.regularizers.l2(wdecay)), 
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D(pool_size=(2, 2)), 
    keras.layers.Dropout(0.3),

    keras.layers.Conv2D(128, kernel_size=3,padding='same', activation='relu',kernel_regularizer=keras.regularizers.l2(wdecay)), 
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(128, kernel_size=3,padding='same', activation='relu',kernel_regularizer=keras.regularizers.l2(wdecay)), 
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D(pool_size=(2, 2)), 
    keras.layers.Dropout(0.4),
    
    keras.layers.Flatten(), 
    keras.layers.Dense(num_classes, activation='softmax'),
])
```

