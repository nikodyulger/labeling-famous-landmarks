from tensorflow import keras
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.cm as cm
import numpy as np
import random
import os


def get_img_array(img_path, size):

    img = keras.preprocessing.image.load_img(img_path, target_size=size)
    array = keras.preprocessing.image.img_to_array(img)
    array = np.expand_dims(array, axis=0)
    return array


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):

    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(
            last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


def save_gradcam(img_path, heatmap, cam_path, alpha):

    img = keras.preprocessing.image.load_img(img_path)
    img = keras.preprocessing.image.img_to_array(img)

    heatmap = np.uint8(255 * heatmap)

    jet = cm.get_cmap("jet")

    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)

    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)

    superimposed_img.save(cam_path)


def get_gradients(img_input, model, top_pred_idx):

    images = tf.cast(img_input, tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(images)
        preds = model(images)
        top_class = preds[:, top_pred_idx]

    grads = tape.gradient(top_class, images)
    return grads


def get_integrated_gradients(img_input, img_size, model, preprocess_input, top_pred_idx, baseline=None, num_steps=50):

    if baseline is None:
        baseline = np.zeros(img_size).astype(np.float32)
    else:
        baseline = baseline.astype(np.float32)

    img_input = img_input.astype(np.float32)
    interpolated_image = [
        baseline + (step / num_steps) * (img_input - baseline)
        for step in range(num_steps + 1)
    ]
    interpolated_image = np.array(interpolated_image).astype(np.float32)

    interpolated_image = preprocess_input(interpolated_image)

    grads = []
    for i, img in enumerate(interpolated_image):
        img = tf.expand_dims(img, axis=0)
        grad = get_gradients(img, model, top_pred_idx=top_pred_idx)
        grads.append(grad[0])
    grads = tf.convert_to_tensor(grads, dtype=tf.float32)

    grads = (grads[:-1] + grads[1:]) / 2.0
    avg_grads = tf.reduce_mean(grads, axis=0)

    integrated_grads = (img_input - baseline) * avg_grads
    return integrated_grads


def random_baseline_integrated_gradients(
    img_input, img_size, model, preprocess_input, top_pred_idx, num_steps=50, num_runs=2, 
):

    integrated_grads = []

    for run in range(num_runs):
        baseline = np.random.random(img_size) * 255
        igrads = get_integrated_gradients(
            img_input=img_input,
            img_size=img_size,
            model= model, 
            preprocess_input=preprocess_input,
            top_pred_idx=top_pred_idx,
            baseline=baseline,
            num_steps=num_steps,
        )
        integrated_grads.append(igrads)

    integrated_grads = tf.convert_to_tensor(integrated_grads)
    return tf.reduce_mean(integrated_grads, axis=0)


def get_random_class_images(class_path, n_images=9, seed=None):

    _, _, filenames = next(os.walk(class_path))
    filenames = [os.path.join(class_path, f) for f in filenames]
    if seed is not None:
        random.seed(seed)
        images_path = random.sample(filenames, n_images)
    else:
        images_path = random.sample(filenames, n_images)

    return images_path


def plot_images_prediction(images_paths, img_size, model, preprocess_input, decode_predictions):

    model.layers[-1].activation = None

    plt.figure(figsize=(10, 10))
    for index, path in enumerate(images_paths):
        ax = plt.subplot(3, 3, index + 1)
        img_array = preprocess_input(get_img_array(path, size=img_size))
        preds = model.predict(img_array)
        dec_preds = decode_predictions(preds, top=1)[0]
        image = mpimg.imread(path)
        plt.imshow(image)
        plt.title(f"Predicted class: [{dec_preds[0][1]}]")
        plt.axis("off")


def plot_gradcam_images(images_paths, img_size, model, preprocess_input, last_conv_layer_name):
    plt.figure(figsize=(12, 12))
    for index, path in enumerate(images_paths):
        ax = plt.subplot(3, 3, index + 1)
        img_array = preprocess_input(get_img_array(path, size=img_size))
        heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)
        save_gradcam(path, heatmap, cam_path=f"./img{index}.jpg", alpha=0.3)
        image = mpimg.imread(f"./img{index}.jpg")
        plt.imshow(image)
        plt.axis("off")
