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
import re
import subprocess
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def model_func(input_vid, output_vid='final_vid.mp4'):
    def infer(img_path1, img_path2, n=4):
        """Run frame interpolation between two images"""
        try:
            model = 'ours_t'
            sys.path.append('.')
            assert model in ['ours_t', 'ours_small_t'], 'Model not exists!'

            TTA = True
            if model == 'ours_small_t':
                TTA = False
                cfg.MODEL_CONFIG['LOGNAME'] = 'ours_small_t'
                cfg.MODEL_CONFIG['MODEL_ARCH'] = cfg.init_model_config(F=16, depth=[2, 2, 2, 2, 2])
            else:
                cfg.MODEL_CONFIG['LOGNAME'] = 'ours_t'
                cfg.MODEL_CONFIG['MODEL_ARCH'] = cfg.init_model_config(F=32, depth=[2, 2, 2, 4, 4])

            model = Model(-1)
            model.load_model()
            model.eval()
            model.device()

            logger.info(f'Starting interpolation between {img_path1} and {img_path2}')

            I0 = cv2.imread(img_path1)
            I2 = cv2.imread(img_path2)

            if I0 is None or I2 is None:
                raise ValueError(f"Could not read one or both input images: {img_path1}, {img_path2}")

            I0_ = (torch.tensor(I0.transpose(2, 0, 1)).cuda() / 255.).unsqueeze(0)
            I2_ = (torch.tensor(I2.transpose(2, 0, 1)).cuda() / 255.).unsqueeze(0)

            padder = InputPadder(I0_.shape, divisor=32)
            I0_, I2_ = padder.pad(I0_, I2_)

            images = [I0[:, :, ::-1]]
            preds = model.multi_inference(I0_, I2_, TTA=TTA, time_list=[(i + 1) * (1. / n) for i in range(n - 1)], fast_TTA=TTA)
            for pred in preds:
                images.append((padder.unpad(pred).detach().cpu().numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8)[:, :, ::-1])
            images.append(I2[:, :, ::-1])
            return images

        except Exception as e:
            logger.error(f"Error during inference: {str(e)}")
            raise

    def extract_frames(video_path, output_folder, image_format="png"):
        """Extract frames from video using ffmpeg"""
        os.makedirs(output_folder, exist_ok=True)
        output_pattern = f"{output_folder}/frame_%04d.{image_format}"
        
        logger.info(f"Extracting frames from {video_path} to {output_folder}")
        
        try:
            # First try with ffmpeg-python
            try:
                (
                    ffmpeg
                    .input(video_path)
                    .output(output_pattern, start_number=0)
                    .run(overwrite_output=True, capture_stdout=True, capture_stderr=True)
                )
                logger.info(f"Frames extracted successfully to {output_folder}")
                return
            except ffmpeg.Error as e:
                logger.warning(f"ffmpeg-python failed, trying direct ffmpeg command. Error: {e.stderr.decode('utf-8') if e.stderr else 'No stderr'}")
            
            # Fallback to subprocess if ffmpeg-python fails
            cmd = [
                'ffmpeg',
                '-i', video_path,
                '-start_number', '0',
                output_pattern
            ]
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info("Frames extracted successfully using subprocess")
            
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg failed with error:\n{e.stderr}")
            raise RuntimeError(f"Failed to extract frames: {e.stderr}")
        except Exception as e:
            logger.error(f"Unexpected error during frame extraction: {str(e)}")
            raise

    def natural_sort_key(filename):
        """Natural sorting for filenames"""
        return [int(num) if num.isdigit() else num.lower() for num in re.split(r'(\d+)', filename)]

    def get_sorted_filenames(folder, extension="png"):
        """Get sorted list of filenames in a folder"""
        filenames = [f for f in os.listdir(folder) if f.endswith(f".{extension}")]
        return sorted(filenames, key=natural_sort_key)

    def compile_video(frames, output_path, fps=30):
        """Compile frames into a video"""
        if not frames:
            raise ValueError("The frame list is empty!")

        first_frame = np.asarray(frames[0])
        height, width, channels = first_frame.shape
        
        logger.info(f"Compiling video {output_path} with {len(frames)} frames at {fps} FPS")
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        for idx, frame in enumerate(frames):
            frame = np.asarray(frame)
            if frame.shape[0] != height or frame.shape[1] != width:
                logger.warning(f"Frame {idx} has size {frame.shape} but expected {height}x{width}")
                frame = cv2.resize(frame, (width, height))
            video_writer.write(frame)

        video_writer.release()
        logger.info(f"Video successfully saved to {output_path}")

    # -----------------------------
    # Main Pipeline Execution
    # -----------------------------
    try:
        # Step 1: Extract frames from input video
        frames_dir = "frames"
        extract_frames(input_vid, frames_dir)
        
        # Verify frame extraction
        filenames = get_sorted_filenames(frames_dir)
        if not filenames:
            raise RuntimeError("No frames were extracted from the video")
        logger.info(f"Extracted {len(filenames)} frames")

        # Step 2: Interpolate between frames using model
        all_frames = []
        for i in range(len(filenames) - 1):
            frame1 = os.path.join(frames_dir, filenames[i])
            frame2 = os.path.join(frames_dir, filenames[i + 1])
            logger.info(f"Interpolating between: {frame1} and {frame2}")
            
            try:
                interpolated = infer(frame1, frame2)
                all_frames.append(interpolated)
            except Exception as e:
                logger.error(f"Failed to interpolate between {frame1} and {frame2}: {str(e)}")
                raise

        # Step 3: Flatten interpolated sequences
        flat_frames = [frame for sublist in all_frames for frame in sublist]
        logger.info(f"Generated {len(flat_frames)} total frames")

        # Step 4: Save interpolated frames
        output_dir = "output_frames"
        os.makedirs(output_dir, exist_ok=True)
        
        for i, img in enumerate(flat_frames):
            if not isinstance(img, np.ndarray):
                logger.warning(f"Skipping frame {i+1}: Invalid type ({type(img)})")
                continue
                
            if img.dtype != np.uint8:
                img = np.clip(img, 0, 255).astype(np.uint8)
                
            filename = os.path.join(output_dir, f"image_{i+1:05d}.png")
            success = cv2.imwrite(filename, img)
            if not success:
                logger.warning(f"Failed to save frame {filename}")

        # Step 5: Compile the enhanced video
        final_filenames = get_sorted_filenames(output_dir)
        if not final_filenames:
            raise RuntimeError("No output frames were generated")
            
        sorted_frames = []
        for filename in final_filenames:
            img = cv2.imread(os.path.join(output_dir, filename))
            if img is None:
                logger.warning(f"Skipping {filename}: Failed to read image")
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            sorted_frames.append(img)

        logger.info(f"Compiling {len(sorted_frames)} frames into final video")
        compile_video(sorted_frames, output_vid, fps=150)

        return output_vid

    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Video Interpolation')
    parser.add_argument('input_vid', type=str, help='Path to the input video file')
    parser.add_argument('--output', type=str, default='final_vid.mp4', help='Output video path')
    args = parser.parse_args()
    
    try:
        result = model_func(args.input_vid, args.output)
        print(f"Successfully created output video: {result}")
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)