import tensorflow as tf
import numpy as np
import cv2
import argparse

def get_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def load_tflite(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    return interpreter, input_details, output_details

def run_inference(interpreter, input_details, output_details, image):
    image_data = image
    input_shape = tuple(input_details[0]['shape'][1:3])
    if input_shape != image.shape[:2]:
        image_data = cv2.resize(np.copy(image), input_shape, interpolation=cv2.INTER_LINEAR)
        # if the shape is different assume that normalization has to be performed as well
        image_data = np.array(image_data)
        image_data = image_data / 255.
    image_data = image_data[np.newaxis, ...].astype(np.float32)

    interpreter.set_tensor(input_details[0]['index'], image_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='TFLite comparison')
    parser.add_argument('--model', type=str,  help="path/to/model.tflite", default="/usr/src/ultralytics/practice_tflite/yolov8n_float32_original.tflite")
    parser.add_argument('--newModel', type=str,  help="path/to/new_model.tflite", default="/usr/src/ultralytics/yolov8n_saved_model/yolov8n_float32.tflite")
    parser.add_argument('--image', type=str,  help="path/to/image.jpg", default="/usr/src/ultralytics/KakaoTalk_Photo_2023-05-23-18-04-12.jpg")
    args = parser.parse_args()

    img = get_image(args.image)
    interpreter_original, input_details_original, output_details_original = load_tflite(args.model)
    interpreter_new, input_details_new, output_details_new = load_tflite(args.newModel)
    output_original = run_inference(interpreter_original, input_details_original, output_details_original, img)
    output_new = run_inference(interpreter_new, input_details_new, output_details_new, img)
    assert np.allclose(output_original, output_new, atol=1e-1) == True




