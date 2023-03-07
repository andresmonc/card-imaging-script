import cv2
import os
import sys

from imutils import contours


def convert(image_file_name, identifier, file_count):
    # Load image
    image = cv2.imread(image_file_name)
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply a threshold to convert the image to black and white
    threshold_value = 242
    max_value = 255
    threshold_type = cv2.THRESH_BINARY_INV
    _, threshold_image = cv2.threshold(gray_image, threshold_value, max_value, threshold_type)
    # Find contours in the thresholded image
    cntrs, hierarchy = cv2.findContours(threshold_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Set the amount of empty space to add around the cropped image
    empty_space = 30

    # Area threshold - ignore small contours with area less than 10000 (assume noise)
    min_contour_area = 50000

    valid_image_counter = file_count

    # clean up contours to filter out garbage
    cntrs = filter_contours_by_area(cntrs, min_contour_area)
    # Sort the contours by their position from top to bottom
    cntrs, _ = contours.sort_contours(cntrs, method="top-to-bottom")

    checkerboard_row = []
    row = []
    for (i, c) in enumerate(cntrs, 1):
        row.append(c)
        if i % 3 == 0:
            (cnts, _) = contours.sort_contours(row, method="left-to-right")
            checkerboard_row.extend(cnts)
            row = []

    cntrs = checkerboard_row
    path = determine_path()
    for i in range(len(cntrs)):
        contour = cntrs[i]
        contour_area = cv2.contourArea(contour)

        # Ignore small contours with area less than min_contour_area (assume noise)
        if contour_area < min_contour_area:
            continue

            # Find the bounding box of the contour
        bounding_rect = cv2.boundingRect(contour)

        # Crop the rectangle from the original image using the bounding box coordinates
        rect = image[bounding_rect[1]:bounding_rect[1] + bounding_rect[3],
               bounding_rect[0]:bounding_rect[0] + bounding_rect[2]].copy()

        # Add empty space around the rect
        rect_with_border = cv2.copyMakeBorder(rect, empty_space * 2, empty_space * 2, empty_space * 2, empty_space * 2,
                                              cv2.BORDER_CONSTANT, value=(255, 255, 255))

        # Save the rectangle with border as a separate image file
        file_name = os.path.join(path, "output", "output_image_{}_{}.png".format(valid_image_counter, identifier))
        cv2.imwrite(file_name, rect_with_border)
        # increase count
        valid_image_counter += 1


def filter_contours_by_area(image_contours, min_contour_area):
    filtered_contours = []
    for contour in image_contours:
        contour_area = cv2.contourArea(contour)
        if contour_area >= min_contour_area:
            filtered_contours.append(contour)
    return filtered_contours


def list_input_files():
    input_files = []
    input_dir = os.path.join(determine_path(), "input")
    files = os.listdir(input_dir)
    for input_file in files:
        if not input_file.startswith('.'):
            input_files.append(input_file)
    num_files = len(input_files)
    if num_files > 2:
        raise ValueError(f"There are {num_files} files in {input_dir}. There should be at most 2 files.")
    return input_files


def list_output_files():
    output_dir = os.path.join(determine_path(), "output")
    files = os.listdir(output_dir)
    return files


def determine_path():
    # determine if application is a script file or frozen exe
    application_path = ""
    if getattr(sys, 'frozen', False):
        application_path = os.path.dirname(sys.executable)
    elif __file__:
        application_path = os.path.dirname(__file__)
    return application_path


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    output_start_index = len(list_output_files()) // 2
    a_or_b = "a"
    app_path = determine_path()
    for file in list_input_files():
        convert(os.path.join(app_path, "input", file), a_or_b, output_start_index)
        a_or_b = "b"
