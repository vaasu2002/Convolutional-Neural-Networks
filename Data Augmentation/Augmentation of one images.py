from skimage import io
from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
        rotation_range=45,     #Random rotation between 0 and 45
        width_shift_range=0.2,   #% shift
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='reflect', cval=125)    #Also try nearest, constant, contant, wrap  -> REFLECT BEST
        
x = io.imread('/content/folder/car_0_1026.jpeg') 
x.shape  # (200, 200, 3)

# We need number of images as first dimension
x = x.reshape((1, ) + x.shape)
x.shape  #  (1,200, 200, 3)      


# Saves 20 images in agumented images folder
i = 0 
for batch in datagen.flow(x, batch_size=16,  
                          save_to_dir='/content/agumented images', 
                          save_prefix='aug', 
                          save_format='png'):
    i += 1
    if i > 20:
        break 
