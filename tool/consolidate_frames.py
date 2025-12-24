import os
import shutil

def consolidate_frames(base_path="."):
    """
    Consolidates all image files from 'frames' to 'frames5' directories
    into a single 'all_frames' directory.

    Args:
        base_path (str): The base directory where 'frames' folders are located.
    """
    output_dir = os.path.join(base_path, "all_frames")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Created output directory: {output_dir}")

    for i in range(6):  # Covers frames, frames1, ..., frames5
        if i == 0:
            source_dir = os.path.join(base_path, "frames")
        else:
            source_dir = os.path.join(base_path, f"frames{i}")

        if not os.path.exists(source_dir):
            print(f"Source directory {source_dir} does not exist. Skipping.")
            continue

        print(f"Processing directory: {source_dir}")
        for filename in os.listdir(source_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                source_file_path = os.path.join(source_dir, filename)
                destination_file_path = os.path.join(output_dir, filename)

                # Handle potential file name conflicts
                counter = 1
                name, ext = os.path.splitext(filename)
                while os.path.exists(destination_file_path):
                    destination_file_path = os.path.join(output_dir, f"{name}_{counter}{ext}")
                    counter += 1

                shutil.move(source_file_path, destination_file_path)
                print(f"Moved '{source_file_path}' to '{destination_file_path}'")
        
        # Optionally, remove the source directory after moving all files
        # if not os.listdir(source_dir): # Check if directory is empty
        #     os.rmdir(source_dir)
        #     print(f"Removed empty source directory: {source_dir}")

if __name__ == "__main__":
    # Call the function to run it directly
    consolidate_frames("C:\\workspace\\Behavior-Data-Analysis-System-Based-on-Deep-Learning")