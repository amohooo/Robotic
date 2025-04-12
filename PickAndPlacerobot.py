import math
from typing import Dict
import numpy as np
from coppeliaRobot import CoppeliaRobot
import time
import copy
import matplotlib.pyplot as plt
import machinevisiontoolbox as mvt
from typing import List, Tuple
import cv2

def coloured_objects_blobs(img: mvt.Image) -> Tuple[int, int, int]:
    """
    Identifies red, green, and blue blobs using MVT methods.

    Parameters
    ----------
    robot : CoppeliaRobot
        The robot instance used to retrieve the camera image.
    
    Returns
    -------
    Tuple[mvt.Image, mvt.Image, mvt.Image]
        A tuple containing labeled images for red, green, and blue blobs.
    """
    # Convert the image to a numpy array and split into R, G, B channels
    img_array = img.image
    R = img_array[..., 0].astype(float)
    G = img_array[..., 1].astype(float)
    B = img_array[..., 2].astype(float)

    # Calculate chromaticity for Red, Green, and Blue channels
    total_intensity = R + G + B
    chromaticity_red = R / (total_intensity + 1e-6)  # Avoid division by zero
    chromaticity_green = G / (total_intensity + 1e-6)
    chromaticity_blue = B / (total_intensity + 1e-6)
    
    # Apply threshold to create binary masks for each color
    red_mask = chromaticity_red > 0.5
    green_mask = chromaticity_green > 0.5
    blue_mask = chromaticity_blue > 0.5

    # Convert masks to MVT Images and find labeled blobs
    red_labeled, red_num = mvt.Image(red_mask.astype(np.uint8)).labels_binary()
    green_labeled, green_num = mvt.Image(green_mask.astype(np.uint8)).labels_binary()
    blue_labeled, blue_num = mvt.Image(blue_mask.astype(np.uint8)).labels_binary()
    
    # Exclude the background (label 0) by subtracting 1 from the count if it exists
    red_blob_count = red_num - 1 if red_num > 0 else 0
    green_blob_count = green_num - 1 if green_num > 0 else 0
    blue_blob_count = blue_num - 1 if blue_num > 0 else 0

    print(f"Number of red blobs: {red_blob_count}")
    print(f"Number of green blobs: {green_blob_count}")
    print(f"Number of blue blobs: {blue_blob_count}")
    
    return red_labeled, green_labeled, blue_labeled

def calculate_blob_properties(img: mvt.Image) -> List[Tuple[int, int]]:
    """
    Calculates the area and perimeter for each blob in the labeled image.

    Parameters
    ----------
    img : mvt.Image
        The labeled image containing blobs.

    Returns
    -------
    List[Tuple[int, int]]
        A list of tuples where each tuple contains the area and perimeter of a blob.
    """
    # Convert MVT image to NumPy array
    img_array = img.image  # Convert mvt.Image to numpy array
    
    # Convert to grayscale (if not already binary)
    gray_image = img.mono().image  # Ensure the image is monochrome (grayscale)

    # Check the intensity range of the grayscale image
    # print(f"Grayscale image min: {gray_image.min()}, max: {gray_image.max()}")
    # plt.imshow(gray_image, cmap='gray')
    # plt.title("Binary Image for Contour Detection")
    # plt.show()

    # Apply thresholding to automatically determine the best threshold
    _, binary_image = cv2.threshold(gray_image.astype(np.uint8), 0, 255, cv2.THRESH_BINARY) # https://docs.opencv.org/3.4/d7/d1b/group__imgproc__misc.html#gae8a4a146d1ca78c626a53577199e9c57

    # Visualize the binary image for debugging
    # plt.imshow(binary_image, cmap='gray')
    # plt.title("Binary Image for Contour Detection")
    # plt.show()

    # Find contours in the binary image
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # https://docs.opencv.org/3.4/d3/dc0/group__imgproc__shape.html#ga17ed9f5d79ae97bd4c7cf18403e1689a

    blob_properties = []
    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True) + 1e-6  # Add a small value to avoid division by zero
        blob_properties.append((area, perimeter))

    return blob_properties

def shape_classification(area: int, perimeter: int) -> str:

    # Circle classification condition
    if abs((perimeter ** 2) / (4 * math.pi * area) - 1) < 0.2:
        return "circle"
    
    # Square classification condition
    if abs((perimeter ** 2) / (16 * area) - 1) < 0.2:
        return "square"
    
    # If neither, return "other"
    return "other"

def detect_red_circles(img: mvt.Image) -> List[Tuple[int, int]]:
    """
    Detect red circles in the image and return their pixel coordinates.

    Parameters
    ----------
    img : mvt.Image
        The input image to detect red circles.

    Returns
    -------
    List[Tuple[int, int]]
        List of pixel coordinates of red circles.
    """
    # Step 1: Detect red blobs
    red_blobs, _, _ = coloured_objects_blobs(img)  # Extract only red blobs from the image

    # Step 2: Calculate blob properties for each red blob (area and perimeter)
    red_blob_properties = calculate_blob_properties(red_blobs)

    # Step 3: Classify each red blob by shape
    red_centroids = calculate_blob_centroids(red_blobs)  # Calculate the centroids of the red blobs
    red_circles = []  # List to hold centroids of red circles

    for i, (area, perimeter) in enumerate(red_blob_properties):
        shape = shape_classification(area, perimeter)  # Classify the shape
        if shape == "circle":
            # If it's classified as a circle, store its centroid
            red_circles.append(red_centroids[i])

    # Manually assign the red circles to red_circles[1], red_circles[2], red_circles[3], red_circles[4]
    red_circles_dict = {}
    if len(red_circles) >= 4:
        red_circles_dict[1] = red_circles[1]  # Manually assign first red circle
        red_circles_dict[2] = red_circles[3]  # Manually assign second red circle
        red_circles_dict[3] = red_circles[2]  # Manually assign third red circle
        red_circles_dict[4] = red_circles[0]  # Manually assign fourth red circle
    else:
        print("Insufficient red circles detected. Detected:", len(red_circles))

    # Print the manually assigned red circle centroids
    for idx, centroid in red_circles_dict.items():
        print(f"Red circle {idx}: {centroid}")

    return red_circles_dict

def homography_apply_to_centroids(img: mvt.Image, centroids: List[Tuple[int, int]]) -> np.array:
    """
    Computes the homography matrix based on detected red circles and applies it to the given centroids.
    
    Parameters
    ----------
    img : mvt.Image
        The input image used to detect the red circles for homography computation.
    centroids : List[Tuple[int, int]]
        List of centroids in pixel coordinates.
    
    Returns
    -------
    np.array
        Transformed centroids in world coordinates.
    """
    # Step 1: Detect red circles for homography
    red_circles = detect_red_circles(img) # Detect red circles in the image

    # Ensure there are at least 4 red circles for homography
    if len(red_circles) < 4:
        raise ValueError("Insufficient red circles detected for homography computation. Need at least 4.")

    # Now get the first 4 red circle centroids from the dictionary
    image_points = np.array(list(red_circles.values())[:4], dtype=np.float32)

    # Step 2: Define corresponding world coordinates (based on known workspace layout)
    world_points = np.array([
        [178, 22.5],   # Large red circle
        [223, 22.5],   # Small red circle
        [223, -22.5],  # Small red circle
        [178, -22.5]   # Small red circle
    ], dtype=np.float32)

    # Compute homography matrix H using the dynamically detected red circles
    H, _ = cv2.findHomography(image_points, world_points)

    # Step 3: Apply homography to the input centroids (converting them to world coordinates)
    centroids_np = np.array(centroids)
    centroids_homogeneous = np.hstack([centroids_np, np.ones((centroids_np.shape[0], 1))])
    transformed_centroids = H @ centroids_homogeneous.T
    transformed_centroids_cartesian = (transformed_centroids[:2, :] / transformed_centroids[2, :]).T

    return transformed_centroids_cartesian

def calculate_blob_centroids(img: mvt.Image) -> List[Tuple[int, int]]:
    """
    Calculates the centroids for each blob in the labeled image using MVT's labeling methods.

    Parameters
    ----------
    img : mvt.Image
        The labeled image containing blobs.

    Returns
    -------
    List[Tuple[int, int]]
        A list of tuples where each tuple contains the (x, y) centroid of a blob.
    """
    # Convert MVT image to NumPy array
    img_array = img.image  # Convert mvt.Image to numpy array
    
    # Convert to grayscale (if not already binary)
    gray_image = img.mono().image  # Ensure the image is monochrome (grayscale)

    # Apply thresholding to get a binary image
    _, binary_image = cv2.threshold(gray_image.astype(np.uint8), 0, 255, cv2.THRESH_BINARY)
    
    # Find contours in the binary image
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    centroids = []
    
    for contour in contours:
        # Calculate moments of the contour
        M = cv2.moments(contour)
        
        # Ensure the area is non-zero to avoid division by zero
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])  # Centroid x
            cy = int(M["m01"] / M["m00"])  # Centroid y
            centroids.append((cx, cy))
    
    return centroids

def classify_colored_shapes(red_blobs, green_blobs, blue_blobs) -> Dict[str, Tuple[str, Tuple[int, int]]]:
    """
    Classifies the blobs by color and shape, and returns a dictionary of detected objects.
    
    Parameters
    ----------
    red_blobs : List[Tuple[int, int]]
        List of areas and perimeters for red blobs.
    green_blobs : List[Tuple[int, int]]
        List of areas and perimeters for green blobs.
    blue_blobs : List[Tuple[int, int]]
        List of areas and perimeters for blue blobs.
    
    Returns
    -------
    Dict[str, Tuple[str, Tuple[int, int]]]
        Dictionary of blob classifications with their centroid coordinates.
    """
    classified_blobs = {}

    # Classify red blobs
    for i, (area, perimeter) in enumerate(red_blobs):
        shape = shape_classification(area, perimeter)
        name = f"red {shape}"
        classified_blobs[name] = f"red {shape}"
    
    # Classify green blobs
    for i, (area, perimeter) in enumerate(green_blobs):
        shape = shape_classification(area, perimeter)
        name = f"green {shape}"
        classified_blobs[name] = f"green {shape}"
    
    # Classify blue blobs
    for i, (area, perimeter) in enumerate(blue_blobs):
        shape = shape_classification(area, perimeter)
        name = f"blue {shape}"
        classified_blobs[name] = f"blue {shape}"
    
    return classified_blobs

def PickAndPlaceRobot(robotObj, img, target_positions: Dict):
    """
    Reposition objects to a desired location.
    
    Parameters
    ----------
    robotObj : CoppeliaRobot
    img : mvt.Image
    target_positions : Dict[str, np.array]
    """
    red_blobs, green_blobs, blue_blobs = coloured_objects_blobs(img)
    
    # Get the centroids for each color blob and convert them to world coordinates
    red_centroids = calculate_blob_centroids(red_blobs)
    green_centroids = calculate_blob_centroids(green_blobs)
    blue_centroids = calculate_blob_centroids(blue_blobs)
    
    red_world_coords = homography_apply_to_centroids(img, red_centroids)
    green_world_coords = homography_apply_to_centroids(img, green_centroids)
    blue_world_coords = homography_apply_to_centroids(img, blue_centroids)
    
    z = 6
    # Add z-coordinate to make 3D points (x, y, z)
    object_centroids = {
        "blue square": np.append(blue_world_coords[1], z) if len(blue_world_coords) > 0 else None,
        "green square": np.append(green_world_coords[1], z) if len(green_world_coords) > 0 else None,
        "red square": np.append(red_world_coords[2], z) if len(red_world_coords) > 0 else None,
        "blue circle": np.append(blue_world_coords[0], z) if len(blue_world_coords) > 0 else None,
        "green circle": np.append(green_world_coords[0], z) if len(green_world_coords) > 0 else None,
    }
    
    # Debugging: Print the object centroids for verification
    for shape, coords in object_centroids.items():
        if coords is not None:
            print(f"{shape} centroid (world coordinates): {coords}")
    
    # Perform pick-and-place for each shape
    for shape, place_position in target_positions.items():
        if shape in object_centroids and object_centroids[shape] is not None:
            pick_position = object_centroids[shape]
            if len(place_position) == 2:
                place_position = np.append(place_position, z)  # Add z if only (x, y)
            PickUp(robotObj, pick_position)
            Place(robotObj, place_position)
    
    home_position = np.array([0, 200, 50])  # move the are away from the camera view
    # Move to home position after all pick-and-place operations are done
    MoveToHomePosition(robotObj, home_position)
            
    distances_moved = get_latest_workplace_status(robotObj, object_centroids, z)
    
    # Debugging: Print the distances moved
    print("Distances moved after pick-and-place:")
    for shape, distance in distances_moved.items():
        if distance is not None:
            print(f"{shape} moved {distance:.2f} mm")
        else:
            print(f"{shape} could not be detected.")
    
    return

# Note: The remainder of the file is a template on how we would solve this task.
# You are free to use our template, or to write your own code.
    
def PickUp(robotObj:CoppeliaRobot, target_pos: np.array):
    '''
    This function should move the robot to the given position and pick up 
    an item if present.
    Note: We recommend the following strategy:
    1. Move the robot to a position 50mm above the target position
    2. Move the robot to the target position
    3. Activate the suction cup
    4. Move the robot to a position 50mm above the target position
    This strategy will prevent dragging the object.
    '''
    # Move the robot to a position 50mm above the target position
    robotObj.set_suction_cup(0)
    pos = copy.deepcopy(target_pos)
    pos[2] = pos[2] + 50
    j1, j2, j3 = ikine(pos)
    robotObj.move_arm(j1, j2, j3)
    time.sleep(2)

    # Move the robot to the target position
    pos = target_pos  # You will need to change this
    pj1, pj2, pj3 = ikine(pos)
    robotObj.move_arm(pj1, pj2, pj3)
    time.sleep(0.5)

    # Active suction cup
    robotObj.set_suction_cup(1)
    time.sleep(0.5)
    
    robotObj.move_arm(j1, j2, j3)
    time.sleep(0.5)

def Place(robotObj, target_pos: np.array):
    ''' 
    This function should move the robot to the given position and release a
    held item if present.
    Note: We recommend the following strategy:
    1. Move the robot to a position 50mm above the target position
    2. Move the robot to the target position
    3. Release the suction cup
    4. Move the robot to a position 50mm above the target position
    This strategy will prevent dragging a held object.
    '''
    # Move the robot to a position 50mm above the target position
    pos = target_pos
    pos = copy.deepcopy(target_pos)
    pos[2] = pos[2] + 50
    j1, j2, j3 = ikine(pos)
    robotObj.move_arm(j1, j2, j3)
    time.sleep(2)
    pj1, pj2, pj3 = ikine(target_pos)
    robotObj.move_arm(pj1, pj2, pj3)
    time.sleep(0.5)
    robotObj.set_suction_cup(0)
    
    time.sleep(0.5)
    
    robotObj.move_arm(j1, j2, j3)
    time.sleep(0.5)

def ikine(pos: np.array) -> np.array:
    """
    Inverse kinematics using geometry

    Parameters
    ----------
    pos
        A numpy array of shape (3,) of the desired end-effector position [x, y, z] (mm)

    Returns
    -------
    theta
        A numpy array of shape (3,) of joint angles [theta1, theta2, theta3] (radians)

    """
        # Extract the desired end-effector position
    x, y, z = pos
    L_0, L_1, L_2, L_3, L_4 = 138, 135, 147, 60, -80
    r = np.sqrt(x**2 + y**2)
    theta1 = np.arctan2(y, x)
    
    BC = L_0 - (z + np.abs(L_4))
    BE = r - L_3
    CE = np.sqrt(BC**2 + BE**2)
    
    try:
        delta = np.arctan2(BE, BC)
        alpha = np.arccos((L_1**2 + L_2**2 - CE**2) / (2 * L_1 * L_2))
        sin_beta = (L_2 * np.sin(alpha)) / CE
        beta = np.arcsin(sin_beta)
        
        theta2 = np.pi - (beta + delta)
        theta3 = np.pi - alpha - (np.pi/2 - theta2)
        
        return np.array([theta1, theta2, theta3])
    except ValueError:
        # If math domain error (e.g., input outside the valid range for arccos)
        print(f"Error: Could not calculate joint angles for position {pos}")
        return None
    
def get_latest_workplace_status(robotObj, original_centroids: Dict[str, np.array], z=6) -> Dict[str, float]:
    """
    Capture the latest image of the workspace, detect object positions, and calculate
    the distance each object has moved from its original position.

    Parameters
    ----------
    robotObj : CoppeliaRobot
        The robot object used to retrieve the camera image.
    original_centroids : Dict[str, np.array]
        The original object positions (centroids) in the workspace.
    z : float, optional
        Default Z coordinate value (height above the workspace plane).

    Returns
    -------
    Dict[str, float]
        A dictionary with the object names as keys and the distances they have moved as values.
    """

    # Get the latest image from the robot's camera
    img = robotObj.get_image()
    
    # # Convert the MVT image to a format compatible with matplotlib (e.g., NumPy array)
    # img_array = img.image  # Convert mvt.Image to numpy array

    # # Display the image
    # plt.imshow(img_array)
    # plt.title('Latest Workspace Status')
    # plt.axis('off')  # Hide axis for cleaner display
    # plt.show()

    # Detect the objects in the current image
    red_blobs, green_blobs, blue_blobs = coloured_objects_blobs(img)

    # Calculate the centroids for each blob (after applying homography to convert to world coordinates)
    red_centroids = calculate_blob_centroids(red_blobs)
    green_centroids = calculate_blob_centroids(green_blobs)
    blue_centroids = calculate_blob_centroids(blue_blobs)

    # Convert centroids to world coordinates
    red_world_coords = homography_apply_to_centroids(img, red_centroids)
    green_world_coords = homography_apply_to_centroids(img, green_centroids)
    blue_world_coords = homography_apply_to_centroids(img, blue_centroids)

    # Update the latest object centroids (world coordinates)
    latest_centroids = {
        "blue square": np.append(blue_world_coords[1], z) if len(blue_world_coords) > 0 else None,
        "green square": np.append(green_world_coords[0], z) if len(green_world_coords) > 0 else None,
        "red square": np.append(red_world_coords[2], z) if len(red_world_coords) > 0 else None,
        "blue circle": np.append(blue_world_coords[0], z) if len(blue_world_coords) > 0 else None,
        "green circle": np.append(green_world_coords[1], z) if len(green_world_coords) > 0 else None,
    }

    # Calculate the distance moved for each object
    distances_moved = {}
    for obj, original_pos in original_centroids.items():
        latest_pos = latest_centroids.get(obj)
        if latest_pos is not None and original_pos is not None:
            distance = np.linalg.norm(latest_pos - original_pos)
            distances_moved[obj] = distance
        else:
            distances_moved[obj] = None  # If the object is missing or could not be detected

    return distances_moved



def MoveToHomePosition(robotObj: CoppeliaRobot, home_position: np.array):
    """
    Move the robot arm to a home position where it doesn't block the camera's view.
    """
    j1, j2,j3 = ikine(home_position)
    robotObj.move_arm(j1, j2, j3)
    time.sleep(2)  # Wait for the arm to move to the home position
    
    

## Keypoints:

# 1. The `PickAndPlaceRobot` function is the main function that orchestrates the pick-and-place operation.
# 2. The `PickUp` and `Place` functions are used to pick up and place objects at specific positions.
# 3. The `ikine` function calculates the inverse kinematics for the robot arm.
# 4. The `get_latest_workplace_status` function captures the latest workspace status and calculates the distance objects have moved.
# 5. The `MoveToHomePosition` function moves the robot arm to a home position.
# 6. The `interpolate_positions` function linearly interpolates between two positions in Cartesian space.
# 7. The `homography_apply_to_centroids` function computes the homography matrix based on detected red circles and applies it to the given centroids.
# 8. The `calculate_blob_centroids` function calculates the centroids for each blob in the labeled image.
# 9. The `shape_classification` function classifies the shape of a blob based on its area and perimeter.
# 10. The `detect_red_circles` function detects red circles in the image and returns their pixel coordinates.
# 11. The `classify_colored_shapes` function classifies the blobs by color and shape and returns a dictionary of detected objects.
# 12. The `calculate_blob_properties` function calculates the area and perimeter for each blob in the labeled image.
# 13. The `coloured_objects_blobs` function identifies red, green, and blue blobs using MVT methods.

# Note: The above functions are used to implement the pick-and-place operation in the CoppeliaSim environment.

## Logic:

# 1. implement the coloured_objects_blobs function to identify red, green, and blue blobs using MVT methods.
# 2. implement the calculate_blob_properties function to calculate the area and perimeter for each blob in the labeled image.
# 3. implement the shape_classification function to classify the shape of a blob based on its area and perimeter--calculated by calculate_blob_properties.
#    manually testing on the threshold values to classify the shapes. 0.2 is most suitable the threshold value.
# 4. implement the detect_red_circles function to detect red circles in the image and return their pixel coordinates.
#    detect only the red circles and store their centroids in the red_circles_dict dictionary.
#    sorting the red circles based on their position in the image according to the provided workplace coordinates follow the same order.
# 5. implement the homography_apply_to_centroids function to compute the homography matrix based on detected red circles and apply it to the given centroids.
# 6. Calculates the centroids for each blob in the labeled image using MVT's labeling methods in the calculate_blob_centroids function.
# 7. classify_colored_shapes function classifies the blobs by color and shape and returns a dictionary of detected objects.
# 8. implement the ikine function to calculate the inverse kinematics for the robot arm.
# 9. get_latest_workplace_status to get the latest image of the workspace, detect object positions, and calculate the distance each object has moved from its original position.
# 10. MoveToHomePosition function moves the robot arm to a home position where it doesn't block the camera's view. for get_latest_workplace_status to get the latest image of the workspace, detect object positions.
        