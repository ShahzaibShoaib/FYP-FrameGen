import cv2
import math
import sys
import torch
import numpy as np
import argparse
from imageio import mimsave
import config as cfg
from Trainer import Model
from benchmark.utils.padder import InputPadder
import ffmpeg
import os

def infer(img_path1,img_path2,n=4):
    
    model = 'ours_t'
    '''==========import from our code=========='''
    sys.path.append('.')
    


    assert model in ['ours_t', 'ours_small_t'], 'Model not exists!'


    '''==========Model setting=========='''
    TTA = True
    if model == 'ours_small_t':
        TTA = False
        cfg.MODEL_CONFIG['LOGNAME'] = 'ours_small_t'
        cfg.MODEL_CONFIG['MODEL_ARCH'] = cfg.init_model_config(
            F = 16,
            depth = [2, 2, 2, 2, 2]
        )
    else:
        cfg.MODEL_CONFIG['LOGNAME'] = 'ours_t'
        cfg.MODEL_CONFIG['MODEL_ARCH'] = cfg.init_model_config(
            F = 32,
            depth = [2, 2, 2, 4, 4]
        )
    model = Model(-1)
    model.load_model()
    model.eval()
    model.device()


    print(f'=========================Start Generating=========================')

    I0 = cv2.imread(img_path1)
    I2 = cv2.imread(img_path2)

    I0_ = (torch.tensor(I0.transpose(2, 0, 1)).cuda() / 255.).unsqueeze(0)
    I2_ = (torch.tensor(I2.transpose(2, 0, 1)).cuda() / 255.).unsqueeze(0)

    padder = InputPadder(I0_.shape, divisor=32)
    I0_, I2_ = padder.pad(I0_, I2_)

    images = [I0[:, :, ::-1]]
    preds = model.multi_inference(I0_, I2_, TTA=TTA, time_list=[(i+1)*(1./n) for i in range(n - 1)], fast_TTA=TTA)
    for pred in preds:
        images.append((padder.unpad(pred).detach().cpu().numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8)[:, :, ::-1])
    images.append(I2[:, :, ::-1])
    #mimsave('example/Capture_out_Nx.gif', images, fps=n)
    # for img in images:
    #     cv2.imshow("Image", img)
    #     cv2.waitKey(0)  # Press any key to show the next image
    #     cv2.destroyAllWindows()
    return images

    #print(f'=========================Done=========================')


def extract_frames(video_path, output_folder, image_format="png"):
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Build the output pattern, e.g., frames/frame_0001.png
    output_pattern = f"{output_folder}/frame_%04d.{image_format}"
    filenames = []
    try:
        (
            ffmpeg
            .input(video_path)
            .output(output_pattern)
            .run(overwrite_output=True)
        )
        print(f"Frames extracted successfully to {output_folder}")
    except ffmpeg.Error as e:
        print("An error occurred while extracting frames:")
        print(e.stderr.decode())
# pop last frame for each iteration ?

def get_sorted_filenames(folder, extension="png"):
    # List all files in the directory that end with the given extension.
    filenames = [f for f in os.listdir(folder) if f.endswith(extension)]
    
    # Sort the filenames lexicographically. This works well for zero-padded names.
    sorted_filenames = sorted(filenames)
    return sorted_filenames


def compile_video(frames, output_path, fps=30):
    """
    Compile a list of frames into a video file.

    Args:
        frames (list): List of frames (images) as numpy arrays or lists convertible to numpy arrays.
        output_path (str): The output video file path (e.g., 'output.mp4').
        fps (int): Frames per second for the output video.
    """
    if not frames:
        raise ValueError("The frame list is empty!")

    # Convert the first frame to a numpy array if needed.
    first_frame = np.asarray(frames[0])
    height, width, channels = first_frame.shape

    # Define the codec and create VideoWriter object.
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for idx, frame in enumerate(frames):
        # Convert frame to numpy array if it's not already one.
        frame = np.asarray(frame)
        if frame.shape[0] != height or frame.shape[1] != width:
            raise ValueError(f"Frame at index {idx} has a different size.")
        video_writer.write(frame)

    video_writer.release()
    print(f"Video successfully saved to {output_path}")



# Example usage:
# image_folder = "frames"
# filenames = get_sorted_filenames(image_folder, "png")
# print(filenames)



# video to images
extract_frames("Apollo_out.mp4", "frames")
filenames = get_sorted_filenames('frames')

# images to interpolated frames
all_frames = []
for i in range(len(filenames)-1):  

    frame1 = 'frames/'+filenames[i]
    frame2 = 'frames/'+filenames[i+1]
    print("FILENAMES ")
    print(frame1)
    print(frame2)
    print()
    all_frames.append(infer(frame1,frame2))




# Flatten the list to get only NumPy arrays
flat_frames = [frame for sublist in all_frames for frame in sublist]

output_dir = "output_frames"
for i, img in enumerate(flat_frames):
    if not isinstance(img, np.ndarray):
        print(f"Skipping frame {i+1}: Invalid type ({type(img)})")
        continue  

    # Convert to uint8 if needed
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)  

    filename = os.path.join(output_dir, f"image_{i+1}.png")
    cv2.imwrite(filename, img)


import re 
def natural_sort_key(filename):
    """Sort filenames in numerical order (image_1.png, image_2.png, ..., image_100.png)"""
    return [int(num) if num.isdigit() else num for num in re.split(r'(\d+)', filename)]

def get_sorted_filenames(folder):
    """Retrieve and sort filenames from a folder"""
    filenames = [f for f in os.listdir(folder) if f.endswith('.png')]
    return sorted(filenames, key=natural_sort_key)


image_folder = "output_frames"
filenames = get_sorted_filenames(image_folder)

data = []
for filename in filenames:
    img = cv2.imread(os.path.join(image_folder, filename))
    if img is None:
        print(f"Skipping {filename}: Failed to read image")
        continue
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    data.append(img)

print("Sorted Filenames:", filenames)

# Compile video
compile_video(data, 'final_vid.mp4', 150)