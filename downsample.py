import ffmpeg

def downsample_video(input_path, output_path, width=1280, height=720, crf=18, preset="slow"):
    (
        ffmpeg
        .input(input_path)
        .output(output_path, vf=f"scale={width}:{height}:flags=lanczos",
                vcodec="libx264", crf=crf, preset=preset,
                acodec="aac", audio_bitrate="128k")
        .run(overwrite_output=True)
    )

# Example usage:
downsample_video("example\output_video_gamma_corrected.mp4", "example\output_video_gamma_corrected_output.mp4")