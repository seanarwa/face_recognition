import argparse
import signal
import sys
import os
import time
import logging as log
from queue import Queue
from threading import Thread
import cv2 as cv
from expiringdict import ExpiringDict

# local modules
import config
import face_detection
import face_encoding
import face_matcher
import announcer

# globals
workers = []
queue = Queue()
tts_cache = ExpiringDict(max_len=100, max_age_seconds=5)
camera = None
matcher = face_matcher.FaceMatcher()


class Worker(Thread):

    exited = False

    def __init__(self, matcher, queue, tts_cache):
        Thread.__init__(self)
        self.matcher = matcher
        self.queue = queue
        self.tts_cache = tts_cache

    def run(self):
        while True:

            if self.exited:
                break

            frames = self.queue.get()
            try:
                faces = face_detection.process(frames)
                for face in faces:
                    save_frame(face)
                    results = self.matcher.match(
                        face_encoding.get_dlib_encodings([face])[0])
                    if len(results) > 0:
                        matched_name = results[0]
                        if matched_name not in self.tts_cache:
                            self.tts_cache[matched_name] = matched_name
                            announcer.say("Welcome, " + str(matched_name))
                    # image_file_name = save_frame(face)
                    # sender.send_frame(frame)
                    # sender.send_image(image_file_name)
            finally:
                self.queue.task_done()

    def kill(self):
        self.exited = True


def print_banner(app_version):
    spaced_text = " FIRM " + str(app_version) + " "
    banner = spaced_text.center(78, '=')
    filler = ''.center(78, '=')
    log.info(filler)
    log.info(banner)
    log.info(filler)


def graceful_shutdown():

    global workers

    log.info('Gracefully shutting down FIRM ...')
    if camera is not None:
        camera.release()
    cv.destroyAllWindows()
    for worker in workers:
        worker.kill()
    sys.exit(0)


def signal_handler(sig, frame):
    log.debug("%s received", signal.Signals(2).name)
    log.debug("Attempting to initiate graceful shutdown ...")
    graceful_shutdown()


def start_capture():

    global queue
    global camera

    spf = 1.00/config.fps

    log.info("Starting capture ...")
    camera = cv.VideoCapture(config.camera_port, cv.CAP_DSHOW)

    while True:

        ret, orig_frame = camera.read()
        if not ret:
            continue

        orig_frame = cv.flip(orig_frame, 1)
        queue.put([orig_frame])

        cv.imshow(config.app_name + " " + config.app_version, orig_frame)

        log.debug("")

        if cv.waitKey(1) == 27:
            graceful_shutdown()

        time.sleep(spf)

    return


def save_frame(frame):
    if frame.size == 0:
        return None
    image_name = str(time.time()) + "." + config.image_type
    path = os.path.join(config.image_output_directory, image_name)
    cv.imwrite(path, frame, config.cv_image_params)
    log.debug("Locally saved %s", image_name)
    log.debug("Image size: %s KB", float(os.stat(path).st_size / 1000))
    return image_name


def main():

    global workers
    global matcher
    global queue
    global tts_cache

    signal.signal(signal.SIGINT, signal_handler)

    # set path to main.py path
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    parser = argparse.ArgumentParser(
        description='Entrypoint script for face-identity-registry-matching (FIRM)'
    )
    parser.add_argument(
        '-f',
        '--config_file',
        help='Path to configuration file.',
        default='config/app.yaml'
    )
    args = parser.parse_args()

    # load app.yaml
    config.load(args.config_file)

    # load matcher
    matcher.load(config.matcher_directory)

    for x in range(100):
        worker = Worker(matcher, queue, tts_cache)
        # setting daemon to True will ignore lifetime of other threads
        worker.daemon = True
        worker.start()
        workers.append(worker)

    print_banner(config.app_version)

    start_capture()


if __name__ == "__main__":
    main()
