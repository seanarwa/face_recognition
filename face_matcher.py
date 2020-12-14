import logging as log
import face_recognition
import os

# local modules
import config

class FaceMatcher:
    names = []
    encodings = []

    def load(self, directory):

        names = []
        file_paths = []

        for dirpath,_,filenames in os.walk(directory):
            for f in filenames:
                names.append(os.path.splitext(f)[0])
                file_paths.append(os.path.abspath(os.path.join(dirpath, f)))
            break

        if len(names) != len(file_paths):
            raise Exception("Matcher names and file paths length mismatch")

        log.info("Loading %d repository file(s) ...", len(file_paths))

        repo_images = []
        for file_path in file_paths:
            repo_images.append(face_recognition.load_image_file(file_path))

        self.names = names
        self.encodings = []
        for repo_image in repo_images:
            self.encodings.append(face_recognition.face_encodings(repo_image)[0])

        log.info("Loaded %d person(s) into matcher", len(names))

    def match(self, encoding):

        matched_names = []

        results = face_recognition.compare_faces(self.encodings, encoding, tolerance=config.matcher_tolerance)
        for name, is_match in zip(self.names, results):
            if is_match:
                matched_names.append(name)

        return matched_names
        


