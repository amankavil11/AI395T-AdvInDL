from pathlib import Path

import fire
from matplotlib import pyplot as plt

from .generate_qa import (
    draw_detections,
    extract_frame_info,
    extract_kart_objects,
    extract_track_info,
)


def generate_caption(info_path: str, view_index: int, img_width: int = 150, img_height: int = 100) -> list:
    """
    Generate caption for a specific view.
    """
    captions = []
    
    # Extract kart objects and track info
    kart_objects = extract_kart_objects(info_path, view_index, img_width, img_height)
    track_name = extract_track_info(info_path)
    
    if not kart_objects:
        return captions
    
    # Find the ego car (track_id == 0)
    ego_car = None
    for kart in kart_objects:
        if kart["is_ego_car"]:
            ego_car = kart
            break
    
    if ego_car is None:
        return captions
    
    ego_center_x, ego_center_y = ego_car["center"]
    ego_kart_name = ego_car["kart_name"]
    
    # 1. Ego car caption
    captions.append(f"{ego_kart_name} is the ego car.")
    
    # 2. Counting caption
    total_karts = len(kart_objects)
    if total_karts == 1:
        captions.append("There is 1 kart in the scene.")
    else:
        captions.append(f"There are {total_karts} karts in the scene.")
    
    # 3. Track name caption
    if track_name:
        captions.append(f"The track is {track_name}.")
    
    # 4. Relative position captions for each kart (except ego car)
    non_ego_karts = [k for k in kart_objects if not k["is_ego_car"]]
    left_count = 0
    right_count = 0
    front_count = 0
    behind_count = 0
    
    for kart in non_ego_karts:
        kart_name = kart["kart_name"]
        kart_center_x, kart_center_y = kart["center"]
        
        # Determine left/right
        is_left = kart_center_x < ego_center_x
        is_right = kart_center_x > ego_center_x
        
        # Determine front/behind (front means lower y value in image coordinates)
        is_front = kart_center_y < ego_center_y
        is_behind = kart_center_y > ego_center_y
        
        if is_left:
            left_count += 1
        if is_right:
            right_count += 1
        if is_front:
            front_count += 1
        if is_behind:
            behind_count += 1
        
        # Build position description
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
            captions.append(f"{kart_name} is {position} of the ego car.")
            # Additional variations
            captions.append(f"The {kart_name} kart is positioned {position} of the ego car.")
            if is_left:
                captions.append(f"{kart_name} is on the left side of the ego car.")
            if is_right:
                captions.append(f"{kart_name} is on the right side of the ego car.")
            if is_front:
                captions.append(f"{kart_name} is ahead of the ego car.")
            if is_behind:
                captions.append(f"{kart_name} is behind the ego car.")
    
    # 5. Additional descriptive captions
    if track_name:
        captions.append(f"This is the {track_name} track.")
        captions.append(f"The scene takes place on the {track_name} track.")
    
    # Counting variations
    if total_karts == 1:
        captions.append("Only the ego car is visible.")
    else:
        captions.append(f"There are {total_karts} karts racing on the track.")
        captions.append(f"The scene shows {total_karts} karts.")
    
    # Position summary captions
    if left_count > 0:
        captions.append(f"There are {left_count} karts to the left of the ego car.")
    if right_count > 0:
        captions.append(f"There are {right_count} karts to the right of the ego car.")
    if front_count > 0:
        captions.append(f"There are {front_count} karts in front of the ego car.")
    if behind_count > 0:
        captions.append(f"There are {behind_count} karts behind the ego car.")
    
    # List all kart names
    all_kart_names = [k["kart_name"] for k in kart_objects]
    captions.append(f"The karts in the scene are: {', '.join(all_kart_names)}.")
    
    # Find closest kart
    if non_ego_karts:
        import numpy as np
        distances = []
        for kart in non_ego_karts:
            kart_center_x, kart_center_y = kart["center"]
            dist = np.sqrt((kart_center_x - ego_center_x)**2 + (kart_center_y - ego_center_y)**2)
            distances.append((dist, kart["kart_name"]))
        
        distances.sort()
        closest_kart = distances[0][1]
        captions.append(f"The closest kart to the ego car is {closest_kart}.")
    
    return captions


def check_caption(info_file: str, view_index: int):
    captions = generate_caption(info_file, view_index)

    print("\nCaption:")
    print("-" * 50)
    for i, caption in enumerate(captions):
        print(f"{i + 1}. {caption}")
        print("-" * 50)

    info_path = Path(info_file)
    base_name = info_path.stem.replace("_info", "")
    image_file = list(info_path.parent.glob(f"{base_name}_{view_index:02d}_im.jpg"))[0]

    annotated_image = draw_detections(str(image_file), info_file)

    plt.figure(figsize=(12, 8))
    plt.imshow(annotated_image)
    plt.axis("off")
    plt.title(f"Frame {extract_frame_info(str(image_file))[0]}, View {view_index}")
    plt.show()


def generate_all_captions(data_dir: str = "data/train", output_file: str = "data/train/balanced_captions.json"):
    """
    Generate captions for all info files in the specified directory.
    
    Args:
        data_dir: Directory containing info.json files
        output_file: Path to output JSON file
    """
    import json
    from PIL import Image
    
    data_path = Path(data_dir)
    all_captions = []
    
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
            
            # Generate captions for this view
            caption_list = generate_caption(str(info_file), view_index, img_width, img_height)
            
            # Add image_file path and each caption as a separate entry
            # Format should be relative to data directory (e.g., "train/00000_00_im.jpg")
            if "train" in str(data_path):
                image_file_path = f"train/{image_file.name}"
            else:
                image_file_path = f"{data_path.name}/{image_file.name}"
            
            # Save each caption as a separate entry
            for caption in caption_list:
                all_captions.append({
                    "image_file": image_file_path,
                    "caption": caption
                })
    
    # Save to JSON file
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(all_captions, f, indent=2)
    
    print(f"Generated {len(all_captions)} caption entries")
    print(f"Saved to {output_path}")


"""
Usage Example: Visualize captions for a specific file and view:
   python -m homework.generate_captions check --info_file data/valid/00000_info.json --view_index 0

Generate captions for all training data:
   python -m homework.generate_captions generate_all --data_dir data/train --output_file data/train/balanced_captions.json
"""


def main():
    fire.Fire({"check": check_caption, "generate_all": generate_all_captions})


if __name__ == "__main__":
    main()
