import os
import shlex
import logging
from subprocess import call
import tempfile
import shutil
import glob
import time
from config_reader import YAMLConfigReader  # Ensure this matches the path to your new module
from logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


class FrameExtractor:
    def __init__(self, video_path: str, fps: int):
        self.video_path = video_path
        # self.tmp_path = f'{tempfile.gettempdir()}/{temp_folder}/'


        config_file = 'params.yml'
        config = YAMLConfigReader(config_file).read_config()
        fps = config['config'].get('FPS', 1)
        temp_dir = config['config']['VIDEO_FRAMES_PATH']

        self.fps = fps
        self.tmp_path = f'{tempfile.gettempdir()}/{temp_dir}/'

    def _create_temp_folder(self, full_path_temp_dir: str):
        if os.path.exists(full_path_temp_dir):
            shutil.rmtree(full_path_temp_dir) 
        os.makedirs(full_path_temp_dir)  # Create a new folder
        logger.info(f"Temporary dir created at: {full_path_temp_dir}")


    def extract_frames(self, start_frame: int, length: float, video_piece_id: int) -> list:
        logger.info(f"Starting frame extraction: start_frame={start_frame}, length={length}, video_piece_id={video_piece_id}")

        self._create_temp_folder(self.tmp_path)

        if not os.access(self.tmp_path, os.W_OK):
            raise Exception(f"{self.tmp_path} is not writable")

        self._run_ffmpeg(start_frame, length, video_piece_id)

        frame_list = self._collect_frame_filenames(length, video_piece_id)
        logger.info(f"Frames extraction completed. Number of frames extracted: {len(frame_list)}")
        
        return frame_list

    def _run_ffmpeg(self, start_frame: int, length: float, video_piece_id: int):
        """Construct and run the ffmpeg command to extract frames."""
        cmd = (
            f"ffmpeg -loglevel panic -ss {start_frame / self.fps} -i {self.video_path} "
            f"-t {length / self.fps} -r {self.fps} -qscale:v 5 {self.tmp_path}{video_piece_id}_%d.jpg"
        )

        if call(shlex.split(cmd), shell=False) != 0:
            logging.critical(f"Failed to extract frames for video: {self.video_path}")
            raise Exception('Failed to extract frames.')
        

    def _collect_frame_filenames(self, length: int, video_piece_id: int) -> list:
        """Collect and return the list of extracted frame filenames."""
        frame_list = []
        _frame_pattern = os.path.join(self.tmp_path, f"{video_piece_id}_*.jpg")
        _frame_files = glob.glob(_frame_pattern)
        _frame_files.sort(key=lambda filename: int(os.path.basename(filename).split('_')[-1].split('.')[0]))

        for frame_file in _frame_files:
            if os.path.exists(frame_file):
                frame_list.append(frame_file)

        return frame_list
    

# if __name__ == "__main__":

#     config_file = 'params.yml'  # Path to your YAML config file
#     video_path = '/home/ajeet/testing/data/output_video_10s.webm'  # Path to the video file

#     config = YAMLConfigReader(config_file).read_config()
#     fps = config['config'].get('FPS', 1)
#     temp_dir = config['config']['VIDEO_FRAMES_PATH']

#     extractor = FrameExtractor(video_path, fps, temp_dir)
#     start_time = time.time()
#     frames = extractor.extract_frames(start_frame=0, length=10, video_piece_id=0)
#     elapsed_time = time.time() - start_time
#     logger.info(f"Time taken to extract {len(frames)} frames: {elapsed_time:.2f} seconds")