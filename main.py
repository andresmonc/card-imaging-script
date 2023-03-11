import cv2
import os
import sys

from imutils import contours


def convert(image_file_name, identifier, file_count):
    # Load Pure Image (never modify)
    image = cv2.imread(image_file_name)
    # try to bright/contrast image
    alpha = 2.5  # Contrast control (1.0-3.0)
    beta = 25  # Brightness control (0-100)
    bright_image = cv2.imread(image_file_name)
    bright_image = cv2.convertScaleAbs(bright_image, alpha=alpha, beta=beta)
    cv2.imwrite("brightness.png", bright_image)
    #

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(bright_image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("gray.png", gray_image)



    # Apply a threshold to convert the image to black and white
    threshold_value = 150
    max_value = 255
    threshold_type = cv2.THRESH_OTSU
    _, threshold_image = cv2.threshold(gray_image, threshold_value, max_value, threshold_type)
    # try adaptive threshold
    # test = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV, 151, 5)
    # cv2.imwrite("test.png", test)
    # threshold_image = test
    #

    # Find contours in the thresholded image
    cntrs, hierarchy = cv2.findContours(threshold_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    cv2.imwrite("thresh.png", threshold_image)
    print(len(cntrs))
    # Set the amount of empty space to add around the cropped image
    empty_space = 30

    # Area threshold - ignore small contours with area less than min_contour_area (assume noise)
    min_contour_area = 300000

    valid_image_counter = file_count

    # clean up contours to filter out garbage
    cntrs = filter_contours_by_area(cntrs, min_contour_area)
    # Sort the contours by their position from top to bottom
    cntrs, _ = contours.sort_contours(cntrs, method="top-to-bottom")

    # identify count of contours per row
    count_per_row = 0
    (_, random_y, _, random_h) = cv2.boundingRect(cntrs[0])
    y_value = random_y + (random_h/2)
    for contour in cntrs:
        (x, y, w, h) = cv2.boundingRect(contour)
        if y <= y_value <= y + h:
            count_per_row += 1

    # Sort the grid of photos left to right
    checkerboard_row = []
    row = []
    for (i, c) in enumerate(cntrs, 1):
        row.append(c)
        if i % count_per_row == 0:
            (cnts, _) = contours.sort_contours(row, method="left-to-right")
            checkerboard_row.extend(cnts)
            row = []
    if len(row) != 0:
        (cnts, _) = contours.sort_contours(row, method="left-to-right")
        checkerboard_row.extend(cnts)

    cntrs = checkerboard_row

    path = determine_path()
    for i in range(len(cntrs)):
        contour = cntrs[i]
        # Find the bounding box of the contour
        bounding_rect = cv2.boundingRect(contour)

        # Crop the rectangle from the original image using the bounding box coordinates
        rect = image[bounding_rect[1]:bounding_rect[1] + bounding_rect[3],
               bounding_rect[0]:bounding_rect[0] + bounding_rect[2]].copy()

        # Add empty space around the rect
        rect_with_border = cv2.copyMakeBorder(rect, empty_space * 2, empty_space * 2, empty_space * 2, empty_space * 2,
                                              cv2.BORDER_CONSTANT, value=(0, 0, 0))

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
