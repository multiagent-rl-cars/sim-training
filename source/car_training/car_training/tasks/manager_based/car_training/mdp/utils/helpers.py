import numpy as np
import cv2
from pathlib import Path
import os
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import logging
PROJECT_ROOT = Path(__file__).parents[8].resolve()
TEST_DATA_PATH = os.path.join(PROJECT_ROOT, "source/car_training/car_training/tasks/manager_based/car_training/mdp/utils/test_data/")

if not os.path.exists(TEST_DATA_PATH):
    os.makedirs(TEST_DATA_PATH)

#### For all of the functions below, we compute reward for only a single copy of the simulation. If we have multiple copies running in parallel, we can call the functions below repeatedly, and that is what we do above. ####

logger = logging.getLogger("car_training" + __name__.rpartition(".car_training")[-1])


def find_track_points_from_image(image: np.ndarray, debug = False, image_idx: int = 0):
    """
    Process the image to find track points.

    :param image: Input image from the car's camera.
    :type image: np.ndarray
    :param debug: If True, will save some debug images. The images will be save in test_data folder.
    :type debug: bool
    :return: List of track points as (x, y) tuples.
    :rtype: list[tuple[float, float]]
    """
    # Convert to grayscale
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    loewr_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 80])

    # Create a binary mask where black pixels are white and the rest are black
    binary = cv2.inRange(hsv, loewr_black, upper_black)

    if debug:
        cv2.imwrite(f"{TEST_DATA_PATH}/debug_binary_{image_idx}.png", binary)
    # Find contours
    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,   # only outer contour
        cv2.CHAIN_APPROX_NONE  # keep all contour points
    )

    # Select the largest contour
    contour = max(contours, key=cv2.contourArea)

    # Extract coordinates
    track_points = contour.squeeze()  # shape (N, 2)

    logger.debug(f"Number of track points detected: {len(track_points)}")

    # Debugging: Save images if debug is True
    if debug:
        debug_contour_image = image.copy()
        for point in track_points:
            cv2.circle(debug_contour_image, (int(point[0]), int(point[1])), 2, (0, 255, 0), -1)
        cv2.imwrite(f"{TEST_DATA_PATH}/debug_track_points_{image_idx}.png", debug_contour_image)

    return track_points

def find_all_tracks(image: np.ndarray, debug = False, image_idx: int = 0):
    """
    Process the image to find track points for all tracks (not just the largest one).

    :param image: Input image from the car's camera.
    :type image: np.ndarray
    :param debug: If True, will save some debug images. The images will be save in test_data folder.
    :type debug: bool
    :return: List of track points as (x, y) tuples.
    :rtype: list[tuple[float, float]]
    """
    # Convert to grayscale
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    loewr_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 80])

    # Create a binary mask where black pixels are white and the rest are black
    binary = cv2.inRange(hsv, loewr_black, upper_black)

    if debug:
        cv2.imwrite(f"{TEST_DATA_PATH}/debug_binary_{image_idx}.png", binary)
    # Find contours
    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_LIST,   # all contours
        cv2.CHAIN_APPROX_NONE  # keep all contour points
    )

    all_track_points = []
    for contour in contours:
        track_points = contour.squeeze()  # shape (N, 2)
        all_track_points.append(track_points)

        logger.debug(f"Number of track points detected in a track: {len(track_points)}")

    # combine all track points into a single list
    all_track_points = np.concatenate(all_track_points, axis=0)

    # Debugging: Save images if debug is True
    if debug:
        debug_contour_image = image.copy()
        for point in all_track_points:
            cv2.circle(debug_contour_image, (int(point[0]), int(point[1])), 2, (0, 255, 0), -1)
        cv2.imwrite(f"{TEST_DATA_PATH}/debug_all_track_points_{image_idx}.png", debug_contour_image)

    return all_track_points

def find_yellow_walls_from_image(image: np.ndarray, debug = False, image_idx: int = 0):
    """
    Process the image to find yellow wall points.

    :param image: Input image from the car's camera.
    :type image: np.ndarray
    :param debug: If True, will save some debug images. The images will be save in test_data folder.
    :type debug: bool
    :return: List of yellow wall points as (x, y) tuples.
    :rtype: list[tuple[float, float]]
    """
    # Convert to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define range for yellow color in HSV
    lower_yellow = np.array([20, 150, 150])
    upper_yellow = np.array([40, 255, 255])

    # Create a mask for yellow color
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    if debug:
        cv2.imwrite(f"{TEST_DATA_PATH}/debug_yellow_mask_{image_idx}.png", mask)

    # Find contours of the masked image
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    contour = max(contours, key=cv2.contourArea)

    yellow_wall_points = contour.squeeze()  # shape (N, 2)

    logger.debug(f"Number of yellow wall points detected: {len(yellow_wall_points)}")

    # if debug draw the yellow wall points on the image
    if debug:
        debug_image = image.copy()
        for point in yellow_wall_points:
            cv2.circle(debug_image, (int(point[0]), int(point[1])), 2, (0, 255, 255), -1)
        cv2.imwrite(f"{TEST_DATA_PATH}/debug_yellow_walls_{image_idx}.png", debug_image)

    return yellow_wall_points

def find_closest_point(
    car_centers: torch.Tensor,        # (N, 2)
    tensor_padded: torch.Tensor,        # (N, T, 2)
    lengths: torch.Tensor              # (N,)
) -> torch.Tensor:
    """
    For each environment, find the closest track point to its car center.

    :param car_centers: Tensor of shape (N, 2) containing the (x, y) coordinates of the car centers for N environments.
    :type car_centers: torch.Tensor
    :param track_padded: Tensor of shape (N, T, 2) containing the padded track points for N environments, where T is the maximum number of track points across all environments.
    :type track_padded: torch.Tensor
    :param lengths: Tensor of shape (N,) containing the actual number of track points for each environment (before padding).
    :type lengths: torch.Tensor
    """
    
    N, Tmax, _ = tensor_padded.shape

    # (N, Tmax)
    dist2 = torch.sum((tensor_padded - car_centers.unsqueeze(1))**2, dim=2)

    # mask padded indices
    idxs = torch.arange(Tmax, device=tensor_padded.device).unsqueeze(0)  # (1, Tmax)
    valid = idxs < lengths.unsqueeze(1)                                 # (N, Tmax)

    dist2 = dist2.masked_fill(~valid, float("inf"))

    # (N,)
    closest_idx = torch.argmin(dist2, dim=1)

    # (N, 2)
    batch = torch.arange(N, device=tensor_padded.device)
    closest_pts = tensor_padded[batch, closest_idx]
    return closest_pts

def get_green_car_mask(image, debug=False):
    """
    Generate a mask for the green car(s) in the image.

    :param image: Input image from the car's camera.
    :type image: np.ndarray

    :return: Binary mask where the green car is detected.
    :rtype: np.ndarray
    """
    # Convert to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define range for green color in HSV
    lower_green = np.array([40, 80, 100])
    upper_green = np.array([120, 255, 255])

    # Create a mask for green color
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Debugging: Save mask image if debug is True
    if debug:
        cv2.imwrite(f"{TEST_DATA_PATH}/debug_green_car_mask.png", mask)

    return mask


def get_green_car_center(image, debug=False):
    """
    Detect the green car in the image and return its center coordinates.

    :param image: Input image from the car's camera.
    :type image: np.ndarray

    :return: Center coordinates of the green car as (x, y) tuple. Returns None if the car is not found.
    :rtype: tuple[float, float] | None
    """
    mask = get_green_car_mask(image, debug=debug)

    # Find contours of the masked image
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Find the largest contour which will correspond to the car
        largest_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest_contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            if debug:
                debug_image = image.copy()
                cv2.circle(debug_image, (cX, cY), 5, (255, 0, 0), -1)
                cv2.imwrite(f"{TEST_DATA_PATH}/debug_green_car_center.png", debug_image)
            return (cX, cY)
    
    return (0,0)  # Car not found


def get_all_car_centers(image, debug=False):
    """
    Detect all green cars in the image and return their center coordinates. This works for multiple agent environments.

    :param image: Input image from the car's camera.
    :type image: np.ndarray

    :return: List of center coordinates of the green cars as (x, y) tuples.
    :rtype: list[tuple[float, float]]
    """
    mask = get_green_car_mask(image, debug=debug)

    # Find contours of the masked image
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    car_centers = []
    for contour in contours:
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            car_centers.append((cX, cY))
            if debug:
                debug_image = image.copy()
                cv2.circle(debug_image, (cX, cY), 5, (255, 0, 0), -1)
                cv2.imwrite(f"{TEST_DATA_PATH}/debug_all_green_car_centers.png", debug_image)

    return car_centers


def convert_coords(
    input_coords: torch.Tensor,
    isaaclab_dimensions: tuple[float, float],
    image_dimensions: tuple[float, float],
    device=None,
    dtype=torch.float32,
    image_to_isaaclab: bool = False,
    preserve_aspect_ratio: bool = True,
) -> torch.Tensor:
    """
    Convert Isaaclab coordinates to image coordinates or vice versa. The default is Isaaclab to image.
    Isaaclab origin is at center, image origin is at top-left.
    Accepts and returns torch tensors.
    """
    # Ensure input is a tensor
    if not torch.is_tensor(input_coords):
        input_coords = torch.tensor(input_coords, dtype=dtype, device=device)

    width_isaaclab, height_isaaclab = isaaclab_dimensions
    width_image, height_image = image_dimensions

    # Unpack coordinates
    x_input = input_coords[..., 0]
    y_input = input_coords[..., 1]

    sx = width_image / width_isaaclab
    sy = height_image / height_isaaclab
    s = min(sx, sy)   # preserve aspect ratio

    pad_x = (width_image - s * width_isaaclab) / 2
    pad_y = (height_image - s * height_isaaclab) / 2

    # Convert coordinates
    if preserve_aspect_ratio:
        if image_to_isaaclab:
            x_output = (x_input - pad_x) / s - (width_isaaclab / 2)
            y_output = (height_isaaclab / 2) - (y_input - pad_y) / s
        else:
            x_output = (x_input + width_isaaclab / 2) * s + pad_x
            y_output = (height_isaaclab / 2 - y_input) * s + pad_y
    else:
        if image_to_isaaclab:
            x_output = (x_input / width_image) * width_isaaclab - (width_isaaclab / 2)
            y_output = (height_isaaclab / 2) - (y_input / height_image) * height_isaaclab
        else:
            x_output = (x_input + width_isaaclab / 2) * sx
            y_output = (height_isaaclab / 2 - y_input) * sy

    # Stack and return as tensor
    output_coords = torch.stack((x_output, y_output), dim=-1)
    if device is not None:
        output_coords = output_coords.to(device)
    return output_coords.type(dtype)

def convert_coords_list(
    input_coords_list: list[tuple[float, float]],
    isaaclab_dimensions: tuple[float, float],
    image_dimensions: tuple[float, float],
    device=None,
    dtype=torch.float32,
    image_to_isaaclab: bool = False,
) -> list[tuple[float, float]]:
    """
    Convert a list of coordinates from Isaaclab to image coordinates or vice versa. The default is Isaaclab to image.
    Isaaclab origin is at center, image origin is at top-left.
    Accepts and returns list of tuples.
    """
    input_coords_tensor = torch.tensor(input_coords_list, dtype=dtype, device=device)
    output_coords_tensor = convert_coords(
        input_coords_tensor,
        isaaclab_dimensions,
        image_dimensions,
        device=device,
        dtype=dtype,
        image_to_isaaclab=image_to_isaaclab,
    )
    output_coords_list = [tuple(coord.tolist()) for coord in output_coords_tensor]
    return output_coords_list


def rotate_torch_image(image: torch.Tensor, angle_degrees: torch.Tensor) -> torch.Tensor:
    """
    Rotate a torch image tensor by a specified angle in degrees.

    :param image: Input grayscale image tensor of shape (N, H, W, 2).
    :type image: torch.Tensor
    :param angle_degrees: Angles to rotate each image in degrees.
    :type angle_degrees: torch.Tensor

    :return: Rotated image tensor of shape (N, 3, H, W).
    :rtype: torch.Tensor
    """
    # TODO: Vectorize this function to avoid the for loop. Maybe need to implement custom rotation function.
    rotated_images = []
    N = image.shape[0]
    for i in range(N):
        img = image[i]  # (3, H, W)
        angle = angle_degrees[i].item()
        rotated_img = TF.rotate(img, angle, expand=False)
        rotated_images.append(rotated_img)
    return torch.stack(rotated_images, dim=0)


def distance_from_other_agents(image, weight: float = 1.0) -> float:
    """
    Function for distances from other agents.

    :param weight: Weight for the penalty.
    :type weight: float

    :return: list of distances from other agents
    :rtype: float
    """
    # TODO: test
    distances = []
    car_centers_in_frame = get_all_car_centers(image)
    for i in car_centers_in_frame:
        for j in car_centers_in_frame:
            if i != j:
                distance = np.linalg.norm(np.array(i) - np.array(j))
                distances.append(distance)
    return distances


def distance_from_other_agents_penalty(image, min_distance: float = 5.0, weight: float = 1.0) -> float:
    """
    Reward function for distances from other agents.

    :param min_distance: Minimum distance to consider for penalty.
    :type min_distance: float
    :param weight: Weight for the penalty.
    :type weight: float

    :return: Penalty based on distances from other agents.
    :rtype: float
    """
    #TODO: min distance needs configuration
    distances = distance_from_other_agents(image, weight=weight)
    penalty = 0.0
    for distance in distances:
        if distance < min_distance:
            penalty += (min_distance - distance) / min_distance * weight
    return penalty



if __name__ == "__main__":
    # test_image = cv2.imread("debug_image_testing.png")
    # car_center = get_green_car_center(test_image, debug=True)
    # print(f"Detected green car center: {car_center}")

    track_images_folder = os.path.join(PROJECT_ROOT, "source/car_training/car_training/tasks/manager_based/car_training/assets/scene_assets/track_images")

    for i in range(12):
        track_img_file = os.path.join(track_images_folder, f"track_{i}.png")
        image = cv2.imread(track_img_file)
        track_points = find_track_points_from_image(image, debug=True, image_name_save=f"debug_track_points_{i}.png")
    # print(f"Number of track points detected: {len(track_points)}")

    # closest_point = __find_closest_track_point(car_center, track_points)
    # print(f"Closest track point to car center: {closest_point}")

    # isaaclab_coords = (2.5, 2.5)
    # isaaclab_dimensions = (10.0, 10.0)  # Example dimensions
    # image_dimensions = (512, 512)  # Example image dimensions
    # image_coords = convert_coords(isaaclab_coords, isaaclab_dimensions, image_dimensions)
    # print(f"Converted image coordinates: {image_coords}")

    # cv2.circle(test_image, (int(closest_point[0]), int(closest_point[1])), 5, (0, 255, 0), -1)
    # cv2.imshow("Car and Closest Track Point", test_image)
    # cv2.waitKey(0)
