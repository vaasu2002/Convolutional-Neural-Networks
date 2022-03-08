# Weather Classification
- Dataset-> https://drive.google.com/drive/folders/1GZGLVJIVlv3lMxJ-sBzzZqWJPrjsnXrx?usp=sharing
- Model->
```ruby
model = Sequential([
    Conv2D(16, (3, 3), activation='relu', input_shape=(256, 256, 3)),
    MaxPooling2D(2, 2),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(256, (3, 3), activation='relu'),
    Conv2D(256, (3, 3), activation='relu'),
    Conv2D(256, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(512, activation='relu'),
    Dense(5, activation='softmax')
])
```
- Image Size for the model -> 256 * 256 
- **IMAGE PRE PROCESSING**
```ruby
def preprocess_image(path):
    img = load_img(path, target_size = (256, 256))
    a = img_to_array(img)
    a = np.expand_dims(a, axis = 0)
    a /= 255.0
    return a
```
