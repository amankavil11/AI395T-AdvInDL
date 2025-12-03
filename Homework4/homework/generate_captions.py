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
