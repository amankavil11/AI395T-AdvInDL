import json
from pathlib import Path

import fire
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

# Define object type mapping
OBJECT_TYPES = {
    1: "Kart",
    2: "Track Boundary",
    3: "Track Element",
    4: "Special Element 1",
    5: "Special Element 2",
    6: "Special Element 3",
}

# Define colors for different object types (RGB format)
COLORS = {
    1: (0, 255, 0),  # Green for karts
    2: (255, 0, 0),  # Blue for track boundaries
    3: (0, 0, 255),  # Red for track elements
    4: (255, 255, 0),  # Cyan for special elements
    5: (255, 0, 255),  # Magenta for special elements
    6: (0, 255, 255),  # Yellow for special elements
}

# Original image dimensions for the bounding box coordinates
ORIGINAL_WIDTH = 600
ORIGINAL_HEIGHT = 400


def extract_frame_info(image_path: str) -> tuple[int, int]:
    """
    Extract frame ID and view index from image filename.

    Args:
        image_path: Path to the image file

    Returns:
        Tuple of (frame_id, view_index)
    """
    filename = Path(image_path).name
    # Format is typically: XXXXX_YY_im.png where XXXXX is frame_id and YY is view_index
    parts = filename.split("_")
    if len(parts) >= 2:
        frame_id = int(parts[0], 16)  # Convert hex to decimal
        view_index = int(parts[1])
        return frame_id, view_index
    return 0, 0  # Default values if parsing fails


def draw_detections(
    image_path: str, info_path: str, font_scale: float = 0.5, thickness: int = 1, min_box_size: int = 5
) -> np.ndarray:
    """
    Draw detection bounding boxes and labels on the image.

    Args:
        image_path: Path to the image file
        info_path: Path to the corresponding info.json file
        font_scale: Scale of the font for labels
        thickness: Thickness of the bounding box lines
        min_box_size: Minimum size for bounding boxes to be drawn

    Returns:
        The annotated image as a numpy array
    """
    # Read the image using PIL
    pil_image = Image.open(image_path)
    if pil_image is None:
        raise ValueError(f"Could not read image at {image_path}")

    # Get image dimensions
    img_width, img_height = pil_image.size

    # Create a drawing context
    draw = ImageDraw.Draw(pil_image)

    # Read the info.json file
    with open(info_path) as f:
        info = json.load(f)

    # Extract frame ID and view index from image filename
    _, view_index = extract_frame_info(image_path)

    # Get the correct detection frame based on view index
    if view_index < len(info["detections"]):
        frame_detections = info["detections"][view_index]
    else:
        print(f"Warning: View index {view_index} out of range for detections")
        return np.array(pil_image)

    # Calculate scaling factors
    scale_x = img_width / ORIGINAL_WIDTH
    scale_y = img_height / ORIGINAL_HEIGHT

    # Draw each detection
    for detection in frame_detections:
        class_id, track_id, x1, y1, x2, y2 = detection
        class_id = int(class_id)
        track_id = int(track_id)

        if class_id != 1:
            continue

        # Scale coordinates to fit the current image size
        x1_scaled = int(x1 * scale_x)
        y1_scaled = int(y1 * scale_y)
        x2_scaled = int(x2 * scale_x)
        y2_scaled = int(y2 * scale_y)

        # Skip if bounding box is too small
        if (x2_scaled - x1_scaled) < min_box_size or (y2_scaled - y1_scaled) < min_box_size:
            continue

        if x2_scaled < 0 or x1_scaled > img_width or y2_scaled < 0 or y1_scaled > img_height:
            continue

        # Get color for this object type
        if track_id == 0:
            color = (255, 0, 0)
        else:
            color = COLORS.get(class_id, (255, 255, 255))

        # Draw bounding box using PIL
        draw.rectangle([(x1_scaled, y1_scaled), (x2_scaled, y2_scaled)], outline=color, width=thickness)

    # Convert PIL image to numpy array for matplotlib
    return np.array(pil_image)


def extract_kart_objects(
    info_path: str, view_index: int, img_width: int = 150, img_height: int = 100, min_box_size: int = 5
) -> list:
    """
    Extract kart objects from the info.json file, including their center points and identify the center kart.
    Filters out karts that are out of sight (outside the image boundaries).

    Args:
        info_path: Path to the corresponding info.json file
        view_index: Index of the view to analyze
        img_width: Width of the image (default: 150)
        img_height: Height of the image (default: 100)

    Returns:
        List of kart objects, each containing:
        - instance_id: The track ID of the kart
        - kart_name: The name of the kart
        - center: (x, y) coordinates of the kart's center
        - is_center_kart: Boolean indicating if this is the kart closest to image center
    """
    with open(info_path) as f:
        info = json.load(f)
    
    karts_list = info["karts"]
    detections = info["detections"]
    
    if view_index >= len(detections):
        return []
    
    frame_detections = detections[view_index]
    
    # Calculate scaling factors
    scale_x = img_width / ORIGINAL_WIDTH
    scale_y = img_height / ORIGINAL_HEIGHT
    
    image_center_x = img_width / 2
    image_center_y = img_height / 2
    
    kart_objects = []
    
    for detection in frame_detections:
        class_id, track_id, x1, y1, x2, y2 = detection
        class_id = int(class_id)
        track_id = int(track_id)
        
        # Only process karts (class_id == 1)
        if class_id != 1:
            continue
        
        # Scale coordinates to fit the current image size
        x1_scaled = x1 * scale_x
        y1_scaled = y1 * scale_y
        x2_scaled = x2 * scale_x
        y2_scaled = y2 * scale_y
        
        # Check if bounding box is too small
        if (x2_scaled - x1_scaled) < min_box_size or (y2_scaled - y1_scaled) < min_box_size:
            continue
        
        # Check if bounding box is within image boundaries
        if x2_scaled < 0 or x1_scaled > img_width or y2_scaled < 0 or y1_scaled > img_height:
            continue
        
        # Calculate center point
        center_x = (x1_scaled + x2_scaled) / 2
        center_y = (y1_scaled + y2_scaled) / 2
        
        # Get kart name
        if track_id < len(karts_list):
            kart_name = karts_list[track_id]
        else:
            continue
        
        kart_objects.append({
            "instance_id": track_id,
            "kart_name": kart_name,
            "center": (center_x, center_y),
            "is_ego_car": (track_id == 0)  # Track ID 0 is the ego car
        })
    
    # Find the kart closest to image center (if no ego car is present)
    if kart_objects:
        min_dist = float('inf')
        center_kart_idx = 0
        for i, kart in enumerate(kart_objects):
            cx, cy = kart["center"]
            dist = np.sqrt((cx - image_center_x)**2 + (cy - image_center_y)**2)
            if dist < min_dist:
                min_dist = dist
                center_kart_idx = i
        
        # Mark the closest kart to center as center kart if it's not the ego car
        for i, kart in enumerate(kart_objects):
            kart["is_center_kart"] = (i == center_kart_idx) and not kart["is_ego_car"]
    
    return kart_objects


def extract_track_info(info_path: str) -> str:
    """
    Extract track information from the info.json file.

    Args:
        info_path: Path to the info.json file

    Returns:
        Track name as a string
    """
    with open(info_path) as f:
        info = json.load(f)
    
    return info.get("track", "")


def generate_qa_pairs(info_path: str, view_index: int, img_width: int = 150, img_height: int = 100) -> list:
    """
    Generate question-answer pairs for a given view.

    Args:
        info_path: Path to the info.json file
        view_index: Index of the view to analyze
        img_width: Width of the image (default: 150)
        img_height: Height of the image (default: 100)

    Returns:
        List of dictionaries, each containing a question and answer
    """
    qa_pairs = []
    
    # Extract kart objects and track info
    kart_objects = extract_kart_objects(info_path, view_index, img_width, img_height)
    track_name = extract_track_info(info_path)
    
    if not kart_objects:
        return qa_pairs
    
    # Find the ego car (track_id == 0)
    ego_car = None
    for kart in kart_objects:
        if kart["is_ego_car"]:
            ego_car = kart
            break
    
    if ego_car is None:
        return qa_pairs
    
    ego_center_x, ego_center_y = ego_car["center"]
    ego_kart_name = ego_car["kart_name"]
    
    # 1. Ego car question
    qa_pairs.append({
        "question": "What kart is the ego car?",
        "answer": ego_kart_name
    })
    
    # 2. Total karts question
    total_karts = len(kart_objects)
    qa_pairs.append({
        "question": "How many karts are there in the scenario?",
        "answer": str(total_karts)
    })
    
    # 3. Track information question
    if track_name:
        qa_pairs.append({
            "question": "What track is this?",
            "answer": track_name
        })
    
    # 4. Relative position questions for each kart (except ego car)
    left_count = 0
    right_count = 0
    front_count = 0
    behind_count = 0
    
    for kart in kart_objects:
        if kart["is_ego_car"]:
            continue
        
        kart_name = kart["kart_name"]
        kart_center_x, kart_center_y = kart["center"]
        
        # Determine left/right
        is_left = kart_center_x < ego_center_x
        is_right = kart_center_x > ego_center_x
        
        # Determine front/behind (front means lower y value in image coordinates)
        is_front = kart_center_y < ego_center_y
        is_behind = kart_center_y > ego_center_y
        
        # Generate relative position questions
        if is_left:
            qa_pairs.append({
                "question": f"Is {kart_name} to the left or right of the ego car?",
                "answer": "left"
            })
            left_count += 1
        elif is_right:
            qa_pairs.append({
                "question": f"Is {kart_name} to the left or right of the ego car?",
                "answer": "right"
            })
            right_count += 1
        
        if is_front:
            qa_pairs.append({
                "question": f"Is {kart_name} in front of or behind the ego car?",
                "answer": "front"
            })
            front_count += 1
        elif is_behind:
            qa_pairs.append({
                "question": f"Is {kart_name} in front of or behind the ego car?",
                "answer": "behind"
            })
            behind_count += 1
        
        # Combined position question
        position_parts = []
        if is_front:
            position_parts.append("front")
        elif is_behind:
            position_parts.append("behind")
        if is_left:
            position_parts.append("left")
        elif is_right:
            position_parts.append("right")
        
        if position_parts:
            position = " and ".join(position_parts)
            qa_pairs.append({
                "question": f"Where is {kart_name} relative to the ego car?",
                "answer": position
            })
    
    # 5. Counting questions
    qa_pairs.append({
        "question": "How many karts are to the left of the ego car?",
        "answer": str(left_count)
    })
    
    qa_pairs.append({
        "question": "How many karts are to the right of the ego car?",
        "answer": str(right_count)
    })
    
    qa_pairs.append({
        "question": "How many karts are in front of the ego car?",
        "answer": str(front_count)
    })
    
    qa_pairs.append({
        "question": "How many karts are behind the ego car?",
        "answer": str(behind_count)
    })
    
    return qa_pairs


def check_qa_pairs(info_file: str, view_index: int):
    """
    Check QA pairs for a specific info file and view index.

    Args:
        info_file: Path to the info.json file
        view_index: Index of the view to analyze
    """
    # Find corresponding image file
    info_path = Path(info_file)
    base_name = info_path.stem.replace("_info", "")
    image_file = list(info_path.parent.glob(f"{base_name}_{view_index:02d}_im.jpg"))[0]

    # Visualize detections
    annotated_image = draw_detections(str(image_file), info_file)

    # Display the image
    plt.figure(figsize=(12, 8))
    plt.imshow(annotated_image)
    plt.axis("off")
    plt.title(f"Frame {extract_frame_info(str(image_file))[0]}, View {view_index}")
    plt.show()

    # Generate QA pairs
    qa_pairs = generate_qa_pairs(info_file, view_index)

    # Print QA pairs
    print("\nQuestion-Answer Pairs:")
    print("-" * 50)
    for qa in qa_pairs:
        print(f"Q: {qa['question']}")
        print(f"A: {qa['answer']}")
        print("-" * 50)


def generate_all_qa_pairs(data_dir: str = "data/train", output_file: str = "data/train/balanced_qa_pairs.json"):
    """
    Generate QA pairs for all info files in the specified directory.
    
    Args:
        data_dir: Directory containing info.json files
        output_file: Path to output JSON file
    """
    from PIL import Image
    
    data_path = Path(data_dir)
    all_qa_pairs = []
    
    # Find all info.json files
    info_files = sorted(data_path.glob("*_info.json"))
    
    print(f"Found {len(info_files)} info files")
    
    for info_file in info_files:
        base_name = info_file.stem.replace("_info", "")
        
        # Find all corresponding image files
        image_files = sorted(info_file.parent.glob(f"{base_name}_*_im.jpg"))
        
        for image_file in image_files:
            # Extract view_index from filename (format: XXXXX_YY_im.jpg)
            parts = image_file.stem.split("_")
            if len(parts) >= 2:
                try:
                    view_index = int(parts[1])
                except ValueError:
                    continue
            else:
                continue
            
            # Get actual image dimensions
            try:
                img = Image.open(image_file)
                img_width, img_height = img.size
            except Exception as e:
                print(f"Warning: Could not read image {image_file}: {e}")
                continue
            
            # Generate QA pairs for this view
            qa_pairs = generate_qa_pairs(str(info_file), view_index, img_width, img_height)
            
            # Add image_file path to each QA pair
            # Format should be relative to data directory (e.g., "train/00000_00_im.jpg")
            if "train" in str(data_path):
                image_file_path = f"train/{image_file.name}"
            else:
                image_file_path = f"{data_path.name}/{image_file.name}"
            
            for qa_pair in qa_pairs:
                qa_pair["image_file"] = image_file_path
                all_qa_pairs.append(qa_pair)
    
    # Save to JSON file
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(all_qa_pairs, f, indent=2)
    
    print(f"Generated {len(all_qa_pairs)} QA pairs")
    print(f"Saved to {output_path}")


"""
Usage Example: Visualize QA pairs for a specific file and view:
   python generate_qa.py check --info_file ../data/valid/00000_info.json --view_index 0

Generate QA pairs for all training data:
   python -m homework.generate_qa generate_all --data_dir data/train --output_file data/train/balanced_qa_pairs.json
"""


def main():
    fire.Fire({"check": check_qa_pairs, "generate_all": generate_all_qa_pairs})


if __name__ == "__main__":
    main()
