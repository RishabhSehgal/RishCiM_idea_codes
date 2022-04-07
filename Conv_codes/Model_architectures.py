import tensorflow as tf
import keras
import math
#import keras_applications
#from pandas import np
from tensorflow.keras.applications import ResNet50, DenseNet121, VGG16, VGG19, EfficientNetV2B0
# example of loading an image with the Keras API
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import array_to_img
#from tensorflow_core.python.keras.utils import plot_model, model_to_dot

#model1 = ResNet50(weights="imagenet", include_top=False)
#model2 = ResNet50(weights="imagenet", include_top=True)

model1 = ResNet50(weights='imagenet', include_top=True)
GAP_output = model1.get_layer('avg_pool').output
model = tf.keras.Model(model1.input, GAP_output)
#plot_model(new_model, to_file='ResNet.png')
#SVG(model_to_dot(new_model).create(prog='dot', format='svg'))
#model2 = ResNet50(include_top=False)
#model2 = VGG16(include_top=False)
model2 = EfficientNetV2B0()

model = DenseNet121()
#model1.summary()
model2.summary()

layer_names = [layer.name for layer in model2.layers]
print(layer_names)
Total_DAC_steps = 0
Total_ADC_steps = 0

for layer in model2.layers:
    # check for convolutional layer
    if "filters" in model2.get_layer(name = layer.name).get_config().keys():

        # summarize input shape
        print(layer.name, layer.input.shape)
        # summarize input shape
        print(layer.name, layer.output.shape)
        # get filter weights
        print(layer.name, layer.weights[0].shape)
        # get strides and padding
        print(layer.name, model2.get_layer(name = layer.name).get_config()['strides'])
        print(layer.name, model2.get_layer(name = layer.name).get_config()['padding'])

        input = layer.input.shape
        output = layer.output.shape
        weights = layer.weights[0].shape
        strides = model2.get_layer(name = layer.name).get_config()['strides']
        padding = model2.get_layer(name = layer.name).get_config()['padding']
        #filters, biases = layer.get_weights()
        #filters = layer.get_weights()
        #print(layer.name, filters.shape)

        ## Calculate the DIM of Im2Col vectors for filters
        #print(layer.weights[0].shape[1])
        if padding == 'valid':
            DimX = math.floor((input[1] + 2*0 - weights[0] + strides[0])/strides[0])
            DimY = math.floor((input[2] + 2*0 - weights[1] + strides[1])/strides[1])
        else:
            DimX = input[1]
            DimY = input[2]

        ## No. of DAC steps
        DAC_steps = DimX * DimY * input[3] * weights[0] * weights[1]
        print("DAC steps", DimX, DimY, DAC_steps)

        ## No. of ADC steps
        ADC_steps = output[1] * output[2] * output[3]
        print("ADC steps", output[1], output[2], output[3], ADC_steps)

        Total_DAC_steps += DAC_steps
        Total_ADC_steps += ADC_steps
        print("Total DAC steps: ", Total_DAC_steps, "Total ADC steps: ", Total_ADC_steps)


#model.layers
layer_outputs = [layer.output for layer in model.layers]
layer_inputs = [layer.input for layer in model.layers]
feature_map_model = tf.keras.models.Model(model.input, layer_outputs)

# load the image
img = load_img("bondi_beach.jpg", target_size=(224, 224))

# report details about the image
print(type(img))
print(img.format)
print(img.mode)
print(img.size)

#img = load_img(image_path, target_size=(150, 150))
input = img_to_array(img)
input = input.reshape((1,) + input.shape)
input /= 255.0

feature_maps = feature_map_model.predict(input)
#for layer_name, feature_map in zip(layer_names, feature_maps):
#    print("The shape of the {$layer_name} is =======>> {$feature_map.shape}")

# summarize feature map size for each conv layer
from keras.applications.vgg16 import VGG16
from matplotlib import pyplot
# load the model
#model = VGG16()
#model = ResNet50()
#model.summary()
# summarize feature map shapes
for i in range(len(model1.layers)):
    layer = model1.layers[i]
    # check for convolutional layer
    if 'conv' not in layer.name:
        continue
    # summarize output shape
    #print(i, layer.name, layer.output.shape)
    # summarize input shape
    #print(i, layer.name, layer.input.shape)
    #summarize weights shape
    # get filter weights
    #filters, biases = layer.get_weights()
    #print(layer.name, filters.shape)



