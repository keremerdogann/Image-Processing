import cv2 as cv2
import shutil
import os
import numpy as np
from moviepy import VideoFileClip

def reverse_video(video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()

    if len(frames) == 0:
        print("Error: The video could not be read or is empty.")
        return

    height, width, layers = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (width, height))

    for frame in reversed(frames):
        out.write(frame)

    out.release()
    print("Video successfully reversed.")

def blur_video(video_path, output_path, kernel_size):
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        blurred_frame = cv2.GaussianBlur(frame, (kernel_size, kernel_size), 0)
        out.write(blurred_frame)

    cap.release()
    out.release()
    print("Video successfully blurred.")

def pan_and_zoom(video_path, output_path, pan_factor):
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        center_x, center_y = width // 2, height // 2
        new_width = int(width * (1 - pan_factor))
        new_height = int(height * (1 - pan_factor))
        x1 = center_x - new_width // 2
        y1 = center_y - new_height // 2
        x2 = center_x + new_width // 2
        y2 = center_y + new_height // 2
        zoomed_frame = cv2.resize(frame[y1:y2, x1:x2], (width, height))
        out.write(zoomed_frame)

    cap.release()
    out.release()
    print("Pan and zoom applied successfully.")

def remove_audio(video_path, output_path):
    clip = VideoFileClip(video_path)
    clip = clip.without_audio()
    clip.write_videofile(output_path, codec="libx264")
    print("Audio removed successfully.")

def add_audio(video_path, audio_path, output_path):
    video = VideoFileClip(video_path)
    audio = VideoFileClip(audio_path).audio
    video = video.set_audio(audio)
    video.write_videofile(output_path, codec="libx264")
    print("Audio added successfully.")

def main():
    video_path = input("Enter the path to the video file: ")
    current_video = "temp_video.mp4"

    if not os.path.exists(video_path):
        print("The video file does not exist.")
        return

    shutil.copy(video_path, current_video)

    while True:
        print("\\nSelect an operation:")
        print("1. Reverse Playback")
        print("2. Blur Video")
        print("3. Pan and Zoom")
        print("4. Remove Audio")
        print("5. Add Audio")
        print("6. Exit and Save")
        choice = input("Enter your choice: ")

        if choice == "1":
            reverse_video(current_video, "reversed.mp4")
            current_video = "reversed.mp4"

        elif choice == "2":
            kernel_size = int(input("Enter blur kernel size (odd number, e.g., 5): "))
            blur_video(current_video, "blurred.mp4", kernel_size)
            current_video = "blurred.mp4"

        elif choice == "3":
            pan_factor = float(input("Enter pan factor (e.g., 0.1 for 10%): "))
            pan_and_zoom(current_video, "panned_zoomed.mp4", pan_factor)
            current_video = "panned_zoomed.mp4"

        elif choice == "4":
            remove_audio(current_video, "no_audio.mp4")
            current_video = "no_audio.mp4"

        elif choice == "5":
            audio_path = input("Enter the path to the audio file: ")
            add_audio(current_video, audio_path, "audio_added.mp4")
            current_video = "audio_added.mp4"

        elif choice == "6":
            output_path = input("Enter the output path to save the video: ")
            shutil.move(current_video, output_path)
            print("Video saved successfully.")
            break

        else:
            print("Invalid choice. Please try again.")

    # C:/Users/Kerem/PycharmProjects/cuda_solution/reklam_videso.mp4

    # C:/Users/Kerem/Desktop


if __name__ == "__main__":
    main()
