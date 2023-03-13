import numpy as np
import pickle
from PIL import Image
from matplotlib import cm

with open('list_happy.pickle', 'rb') as f:
    neural_data_happy = pickle.load(f)

n = 0

for i in neural_data_happy[0:30001]:
    image = Image.fromarray(np.uint8(cm.gist_earth(i) * 255))
    # PIL_image = Image.fromarray(np.uint8(numpy_image)).convert('RGB')

    right = 33
    left = 33
    top = 14
    bottom = 14
    width, height = image.size

    new_width = width + right + left
    new_height = height + top + bottom
    result = Image.new('RGB', (new_width, new_height), (255, 255, 255))
    result.paste(image, (left, top))
    result.save(f'/home/u956278/openai/openai_happy/neural_imgs_happy/im_{n}.png')
    n += 1
