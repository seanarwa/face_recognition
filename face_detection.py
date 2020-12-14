import os
import time
import logging as log
import cv2 as cv
import numpy as np
import face_recognition

# local modules
import config


def process(frames):

    log.debug("Processing frame ...")
    log.debug("Extraction layers: %s", config.extraction_layers)
    total_exec_time = 0.0

    layer_count = 1
    for extraction_layer in config.extraction_layers:

        start_exec_time = time.time()

        if extraction_layer == "haarcascade":
            frames = extract_haarcascade_faces(frames)
        elif extraction_layer == "caffemodel":
            frames = extract_caffemodel_faces(frames)
        elif extraction_layer == "dlib":
            frames = extract_dlib_faces(frames)
        else:
            log.error("Unknown extraction layer: %s", extraction_layer)
            exit(0)

        exec_time = time.time() - start_exec_time

        log.debug("Extraction layer %d - %s execution time: %s",
                  layer_count, extraction_layer, exec_time)
        log.debug("Extraction layer %d - %s captured: %d",
                  layer_count, extraction_layer, len(frames))
        layer_count += 1
        total_exec_time += exec_time

        if len(frames) == 0:
            return []

    frames = [frame for frame in frames if len(frame) != 0]

    log.debug("Total face(s) extracted: %s", len(frames))
    log.debug("Total execution time: %s", total_exec_time)

    return frames


def extract_haarcascade_faces(frames):

    results = []

    for frame in frames:

        if frame.size == 0:
            continue

        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        frame_gray = cv.equalizeHist(frame_gray)
        faces = config.haarcascade_face_cascade.detectMultiScale(
            frame_gray,
            scaleFactor=1.4,
            minNeighbors=1,
            minSize=(30, 30)
        )

        for (x, y, w, h) in faces:

            cropped_img = frame[y:y+h, x:x+w]
            results.append(cropped_img)

            # faceROI = frame_gray[y:y+h,x:x+w]
            # eyes = config.haarcascade_eyes_cascade.detectMultiScale(faceROI)
            # noses = config.haarcascade_nose_cascade.detectMultiScale(faceROI)
            # mouths = config.haarcascade_mouth_cascade.detectMultiScale(faceROI)
            # if (len(eyes) == 2):
            #     cropped_img = frame[y:y+h, x:x+w]
            #     results.append(cropped_img)

    return results


def extract_caffemodel_faces(frames):

    results = []

    for frame in frames:

        if frame.size == 0:
            continue

        (h, w) = frame.shape[:2]
        blob = cv.dnn.blobFromImage(cv.resize(frame, (300, 300)), 1.0,
                                    (300, 300), (104.0, 177.0, 123.0))

        config.caffemodel_net.setInput(blob)
        detections = config.caffemodel_net.forward()

        for i in range(0, detections.shape[2]):

            confidence = detections[0, 0, i, 2]

            if confidence < config.caffemodel_confidence_threshold:
                continue

            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            y = startY - 10 if startY - 10 > 10 else startY + 10

            cropped_img = frame[startY:endY, startX:endX]
            results.append(cropped_img)

    return results


def extract_dlib_faces(frames):

    results = []

    for frame in frames:

        if frame.size == 0:
            continue

        rgb_small_frame = None

        if config.dlib_frame_resize_enabled:
            resize_scale = config.dlib_frame_resize_scale
            small_frame = cv.resize(
                frame, (0, 0), fx=resize_scale, fy=resize_scale)
            rgb_small_frame = small_frame[:, :, ::-1]
        else:
            rgb_small_frame = frame[:, :, ::-1]

        face_locations = face_recognition.face_locations(
            rgb_small_frame, number_of_times_to_upsample=config.dlib_upsamples, model=config.dlib_model)

        for (top, right, bottom, left) in face_locations:

            if config.dlib_frame_resize_enabled:
                inverted_scale = int(1 / config.dlib_frame_resize_scale)
                top *= inverted_scale
                right *= inverted_scale
                bottom *= inverted_scale
                left *= inverted_scale

            cropped_img = frame[top:bottom, left:right]
            results.append(cropped_img)

    return results
