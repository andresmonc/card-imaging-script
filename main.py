import cv2
import os
import sys
import statistics

from imutils import contours


def convert(image_file_name, identifier, file_count, attempt, expected_card_count):
    # settings
    max_attempt_count = 3
    print("attempt count: " + str(attempt))

    if attempt > max_attempt_count:
        print("fatal error")
        exit(1)
    # Load Pure Image (never modify)
    image = cv2.imread(image_file_name)

    # try to bright/contrast image
    modified_image = dynamic_image_modifier(image_file_name, attempt)

    # Convert the image to grayscale
    gray_image = to_grayscale(modified_image)

    threshold_image = dynamic_thresholder(gray_image, attempt)

    # get contours from a threshold image
    cntrs = get_contours(threshold_image)
    # sort contours top to bottom left to right
    cntrs = sort_contours(cntrs)

    # determine output path based on OS
    path = determine_path()
    # Track valid image outputs
    valid_image_counter = file_count
    # Set the amount of empty space to add around the cropped image
    empty_space = 30

    if contour_error(cntrs, expected_card_count):
        print("Failure - attempting new configuration")
        return convert(image_file_name, identifier, file_count, attempt + 1, expected_card_count)

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


def to_grayscale(image_to_grayscale):
    gray_image = cv2.cvtColor(image_to_grayscale, cv2.COLOR_BGR2GRAY)
    # cv2.imwrite("gray.png", gray_image)
    return gray_image


def dynamic_image_modifier(image_to_modify_name, image_modify_attempt_count):
    brightnesses = [-10, -10, -10, 20]
    alpha = 2.0  # Contrast control (1.0-3.0)
    beta = brightnesses[image_modify_attempt_count]  # Brightness control (0-100)
    modified_image = cv2.convertScaleAbs(cv2.imread(image_to_modify_name), alpha=alpha, beta=beta)
    # cv2.imwrite("brightness.png", modified_image)
    if image_modify_attempt_count == 0:
        bitwise = cv2.bitwise_not(modified_image)
        # cv2.imwrite("bitwise.png", bitwise)
        modified_image = bitwise
    return modified_image


def dynamic_thresholder(gray_image_to_threshold, thresholding_attempt_count):
    # Apply a threshold to convert the image to black and white
    threshold_value = 241
    max_value = 255
    threshold_image = []
    if thresholding_attempt_count == 0:
        _, threshold_image = cv2.threshold(gray_image_to_threshold, threshold_value, max_value, cv2.THRESH_BINARY_INV)
    elif thresholding_attempt_count == 1:
        _, threshold_image = cv2.threshold(gray_image_to_threshold, threshold_value, max_value, cv2.THRESH_OTSU)
    elif thresholding_attempt_count == 2:
        threshold_image = cv2.adaptiveThreshold(gray_image_to_threshold, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                cv2.THRESH_BINARY_INV, 11, 2)
    elif thresholding_attempt_count == 3:
        threshold_image = cv2.adaptiveThreshold(gray_image_to_threshold, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                                cv2.THRESH_BINARY_INV, 21, 2)

    # cv2.imwrite("threshold.png", threshold_image)
    return threshold_image


def contour_error(contours_to_check, expected_card_count_to_check):
    if len(contours_to_check) != expected_card_count_to_check:
        return True
    # don't use this program for 1 picture lol
    if len(contours_to_check) <= 1:
        return True
    allowed_standard_deviation = 30000
    contour_areas = []
    for i in range(len(contours_to_check)):
        contour_areas.append(cv2.contourArea(contours_to_check[i]))
    std_dev = statistics.stdev(contour_areas)
    if std_dev > allowed_standard_deviation:
        return True
    return False


def sort_contours(contours_to_sort):
    # Sort the contours by their position from top to bottom
    cntrs, _ = contours.sort_contours(contours_to_sort, method="top-to-bottom")

    # identify count of contours per row
    count_per_row = 0
    (_, random_y, _, random_h) = cv2.boundingRect(cntrs[0])
    y_value = random_y + (random_h / 2)
    for contour in cntrs:
        (x, y, w, h) = cv2.boundingRect(contour)
        if y <= y_value <= y + h:
            count_per_row += 1

    # Sort the grid of photos left to right
    checkerboard_sorted_contours = []
    row = []

    for (i, c) in enumerate(cntrs, 1):
        row.append(c)
        if i % count_per_row == 0:
            (cnts, _) = contours.sort_contours(row, method="left-to-right")
            checkerboard_sorted_contours.extend(cnts)
            row = []
    if len(row) != 0:
        (cnts, _) = contours.sort_contours(row, method="left-to-right")
        checkerboard_sorted_contours.extend(cnts)

    return checkerboard_sorted_contours


def get_contours(threshold):
    # Find contours in the thresholded image
    cntrs, hierarchy = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.imwrite("thresh.png", threshold)

    # Area threshold - ignore small contours with area less than min_contour_area (assume noise)
    min_contour_area = 420000

    # clean up contours to filter out garbage
    return filter_contours_by_area(cntrs, min_contour_area)


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
    print("""
  _____                   _              _           _____                      _____                                              
 |  __ \                 | |            ( )         |  __ \                    / ____|                                             
 | |  | |  _   _    ___  | | __  _   _  |/   ___    | |__) |  _ __    ___     | (___     ___    __ _   _ __    _ __     ___   _ __ 
 | |  | | | | | |  / __| | |/ / | | | |     / __|   |  ___/  | '__|  / _ \     \___ \   / __|  / _` | | '_ \  | '_ \   / _ \ | '__|
 | |__| | | |_| | | (__  |   <  | |_| |     \__ \   | |      | |    | (_) |    ____) | | (__  | (_| | | | | | | | | | |  __/ | |   
 |_____/   \__,_|  \___| |_|\_\  \__, |     |___/   |_|      |_|     \___/    |_____/   \___|  \__,_| |_| |_| |_| |_|  \___| |_|   
                                  __/ |                                                                                            
                                 |___/                                                                                             
    """)
    print("Written by Jaime Moncayo v1.3")
    card_count = input("Card Count Per Image: ")
    output_start_index = len(list_output_files()) // 2
    a_or_b = "a"
    app_path = determine_path()
    for file in list_input_files():
        convert(os.path.join(app_path, "input", file), a_or_b, output_start_index, 0, int(card_count))
        a_or_b = "b"
    print("Success!!!")
    input("Press enter to exit...")
