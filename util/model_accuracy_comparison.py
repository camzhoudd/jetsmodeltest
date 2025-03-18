import time
import coremltools as ct
from pathlib import Path
import numpy as np
from PIL import Image


def process_file(model_1, model_2, input_file, coordinate_differences, confidence_differences, speed_diff_1, speed_diff_2, anomalies):
    image_path = input_file  # Replace with your image file path
    image = Image.open(image_path)
    resizeimage = image.resize((640, 640))

    my_dict = {
        'iouThreshold': 0.45,
        'image': resizeimage,
        'confidenceThreshold': 0.35
    }

    # Get output predictions for both models
    start_time = time.time()
    output_1 = model_1.predict(my_dict)
    mid = time.time()
    output_2 = model_2.predict(my_dict)
    end = time.time()

    execution_time1 = end - start_time
    execution_time2 = end - mid
    speed_diff_1.append(execution_time1)
    speed_diff_2.append(execution_time2)

    coordinates_1 = output_1['coordinates']
    confidence_1 = output_1['confidence']

    coordinates_2 = output_2['coordinates']
    confidence_2 = output_2['confidence']

    coordinate_difference = np.abs(coordinates_1 - coordinates_2)
    confidence_difference = np.abs(confidence_1 - confidence_2)

    if len(coordinate_difference) > 0:
        coordinate_differences.append(np.max(coordinate_difference))
    if len(confidence_difference) > 0:
        confidence_differences.append(np.max(confidence_difference))

    if (execution_time1 > 0.05 or execution_time2 > 0.05):
        anomalies.append(input_file)

def five_stat_analysis(data):
    # Sort the data first
    data_sorted = np.sort(data)

    # 5-number summary
    minimum = np.min(data_sorted)
    q1 = np.percentile(data_sorted, 25)
    median = np.median(data_sorted)
    q3 = np.percentile(data_sorted, 75)
    maximum = np.max(data_sorted)

    # Print the 5-number summary
    print(f"Minimum: {minimum}")
    print(f"First Quartile (Q1): {q1}")
    print(f"Median (Q2): {median}")
    print(f"Third Quartile (Q3): {q3}")
    print(f"Maximum: {maximum}")


def main():
    model_1_path = 'TagYOLOModel_16.mlmodel'  
    model_2_path = 'TagYOLOModel.mlmodel'  

    model_1 = ct.models.MLModel(model_1_path)
    model_2 = ct.models.MLModel(model_2_path)

    speed_diff_1 = []
    speed_diff_2 = []
    coordinate_differences = []
    confidence_differences = []
    anomalies = []

    # Specify the directory
    directory = Path("hannaford/")

    # Loop through the files in the directory
    for file_path in directory.iterdir():
        if file_path.is_file():
            print(f"Processing file: {file_path}")
            if file_path.suffix.lower() == '.png':
                process_file(model_1, model_2, file_path, coordinate_differences, confidence_differences, speed_diff_1, speed_diff_2, anomalies)

    print("==== Speed Diff 1 =====")
    five_stat_analysis(speed_diff_1)
    print("==== Speed Diff 2 =====")
    five_stat_analysis(speed_diff_2)
    print("==== Coordinate Diff 1 =====")
    five_stat_analysis(coordinate_differences)
    print("==== Confidence Diff 1 =====")
    five_stat_analysis(confidence_differences)
    print("==== Anomalies =====")
    print(anomalies)

if __name__ == "__main__":
    main()
