import os
import cv2
import shlex
import logging
from subprocess import call
import yaml
import tempfile

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler()],
    level=logging.DEBUG
)
logger = logging.getLogger(__name__)

class FrameExtractor:
    def __init__(self, session_video_path, config):
        """
        Initialize FrameExtractor with video path and configuration settings.
        :param session_video_path: Path to the video file
        :param config: Configuration settings loaded from YAML file
        """
        self.session_video_path = session_video_path
        self.tmp_path = config['config']['VIDEO_FRAMES_PATH']
        self.fps = config['config'].get('FPS_VIDEO', 1) # Default to 1 if FPS_VIDEO is not specified

        self.tmp_path = f'{tempfile.gettempdir()}/{self.tmp_path}/' 
        self.create_temp_path()

    def create_temp_path(self):
        """Create the temporary path if it doesn't exist."""

        os.makedirs(self.tmp_path, exist_ok=True)  # Create directory if it doesn't exist
        logger.info(f"Temporary path created at: {self.tmp_path}")

    def extract_frames(self, start_frame, length, video_piece_id):
        """
        Extract frames from a specific video chunk.

        :param start_frame: Index of the start of the chunk
        :param length: Length of the chunk in frames
        :param video_piece_id: Identifier for the video chunk
        :return: List of extracted frame filenames
        """
        logger.info(f"Starting frame extraction: start_frame={start_frame}, length={length}, video_piece_id={video_piece_id}")

        # Ensure temp path is writable
        if not os.access(self.tmp_path, os.W_OK):
            raise Exception(f"{self.tmp_path} is not writable")

        # Delete existing images
        self._delete_existing_images(video_piece_id)

        # Construct and execute the ffmpeg command
        self._run_ffmpeg(start_frame, length, video_piece_id)

        # Collect and return the list of frame filenames
        frame_list = self._collect_frame_filenames(video_piece_id, length)
        logger.info(f"Extraction completed. Number of frames extracted: {len(frame_list)}")
        
        return frame_list

    def _delete_existing_images(self, video_piece_id):
        """Delete existing frame images based on video_piece_id."""
        all_img_filenames = os.path.join(self.tmp_path, f"{video_piece_id}_*.jpg")
        logging.info(f"Deleting existing images: {all_img_filenames}")
        call(shlex.split(f"rm -rf {all_img_filenames}"), shell=False)

    def _run_ffmpeg(self, start_frame, length, video_piece_id):
        """Construct and run the ffmpeg command to extract frames."""
        cmd = (
            f"ffmpeg -loglevel panic -ss {start_frame / self.fps} -i {self.session_video_path} "
            f"-t {length / self.fps} -r {self.fps} -qscale:v 5 {self.tmp_path}{video_piece_id}_%d.jpg"
        )
        logging.info(f"Running ffmpeg command:")

        if call(shlex.split(cmd), shell=False) != 0:
            raise Exception('Failed to run ffmpeg.')

    def _collect_frame_filenames(self, video_piece_id, length):
        """Collect and return the list of extracted frame filenames."""
        frame_list = []
        _frame_pattern = os.path.join(self.tmp_path, f"{video_piece_id}_*.jpg")

        for frame_id in range(1, num_img_files_extracted + 1):
            filename = os.path.join(self.tmp_path, f"{video_piece_id}_{frame_id}.jpg")
            if os.path.exists(filename):
                frame_list.append(filename)

        return frame_list
    
def load_config(config_path):
    """
    Load the YAML configuration file.
    :param config_path: Path to the YAML configuration file
    :return: Dictionary containing configuration settings
    """

    parameter_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'params.yml')

    with open(parameter_file, 'r') as config_file:
        config = yaml.safe_load(config_file)
    return config


if __name__ == "__main__":  
    session_video_path = '/home/ajeet/testing/data/output_video_10s.webm'
    config_path = 'params.yml'
    config = load_config(config_path)
    extractor = FrameExtractor(session_video_path, config)
    frames = extractor.extract_frames(start_frame=0, length=10, video_piece_id=1)
    logging.info("Extracted frames: %s", frames)







    # def generate_frames(self, start_frame=None, length=None, video_piece_id=0):
    #     """
    #     This function extract frames belonging to a particular video chunk. Enhances the contrast of
    #     images, reads them into the memory and deletes from the tempporary path.
    #     :param start_frame: index of the start of chunk
    #     :param length: length of the chunk
    #     :return: list of images (data type: list of numpy.ndarray elements)
    #     """
    #     # if fps is 0:
    #     #     fps = self.fps
    #     if self.settings['ENABLE_PERSON_SEGMENTER']:
    #         # fps = 9
    #         fps = 1
    #     else:
    #         # fps = 3
    #         fps = 1

    #     tmp = self.settings['ALL_TEMP_PATH']
    #     video_filename = os.path.basename(self.session_video_path)

    #     frame_list = []
    #     for frame_num in range(start_frame, start_frame + length):
    #         filename = tmp + str(video_piece_id) + '_' + str(frame_num) + '.png'
    #         if os.path.exists(filename):
    #             frame = cv2.imread(filename)
    #             img_yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
    #             # equalize the histogram of the Y channel
    #             img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
    #             # convert the YUV image back to RGB format
    #             frame = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    #             frame_list.append(frame)
    #     if frame_list:
    #         return frame_list

    #     if not os.access(tmp, os.W_OK):
    #         raise Exception(tmp + " is not writable")
    #         return 0

    #     # delete existing images
    #     all_img_filenames = tmp + str(video_piece_id) + '_*.jpg'
    #     call(shlex.split("rm " + all_img_filenames), shell=False)

    #     if start_frame is not None and length is not None:
    #         if self.settings['ENABLE_PERSON_SEGMENTER']:
    #             cmd = "ffmpeg -loglevel panic -ss " + str(3 * start_frame / fps) + " -i " + self.session_video_path + " -t " \
    #                   + str((3 * length) / fps) + " -r " + str(fps) + \
    #                   " -qscale:v 5 " + tmp + str(video_piece_id) + '_%d.jpg'
    #         else:
    #             cmd = "ffmpeg -loglevel panic -ss " + str(start_frame / fps) + " -i " + self.session_video_path + " -t " \
    #               + str(length / fps) + " -r " + str(fps) + \
    #               " -qscale:v 5 " + tmp + str(video_piece_id) + '_%d.jpg'
    #     else:
    #         cmd = "ffmpeg -i " + self.session_video_path + " -r " + \
    #               str(fps) + " -qscale:v 5 " + tmp + str(video_piece_id) + '_%d.jpg'

    #     runcmd = call(shlex.split(cmd), shell=False)

    #     if runcmd != 0:
    #         raise Exception('failed to run ffmpeg. Possible reasons: ffmpeg not in path, or disk is full, or '
    #                         + self.session_video_path + ' is not accessible')
    #         return 0

    #     frame_list = []
    #     frameid = 1  # first frame is 0, but 1 in disk, so '+1'
    #     if self.settings['ENABLE_PERSON_SEGMENTER']:
    #         num_img_files_extracted = int(length * 3)
    #     else:
    #         num_img_files_extracted = int(length)
    #     # logging.info("TestSession {0} | Start {1}. Length {2}. Num_imgs: {3}"
    #     #              .format(self.settings['TESTSESSION_ID'], start_frame, length, num_img_files_extracted))
    #     while frameid <= num_img_files_extracted:
    #         filename = tmp + str(video_piece_id) + '_' + str(frameid) + '.jpg'
    #         # frame = cv2.imread(filename)
    #         # if frame is None:
    #         #     break
    #         if os.path.exists(filename):
    #             frame_list.append(filename)

    #         frameid += 1

    #     return frame_list

    # # based on current settings,
    # # generate a list,
    # # list = [(start1, length1), (start2, length2) ...]
    # # each element in the list the video pieces
    # # THE FIRST FRAME IS FRAME #0