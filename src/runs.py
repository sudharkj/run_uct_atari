import cv2
# import numpy as np
import os
import pickle
import tarfile


def resize(frame, **kwargs):
    width = kwargs['width']
    height = kwargs['height']
    downsample = kwargs['downsample']

    h, w = frame.shape[:2]

    if downsample > 0:
        width = int(w / downsample)
        height = int(h / downsample)
    else:
        downsample = 0.5 * w / width + 0.5 * h / height

    if downsample > 4:
        frame = cv2.resize(frame, (width * 2, height * 2))
    return cv2.resize(frame, (width, height))


def preprocess_run(run, **kwargs):
    if run['reward'] >= kwargs['min_score']:
        run['frames'] = [resize(frame, **kwargs) for frame in run['frames']]


def is_not_pickle_file(file_name):
    # true if the file name end with .pkl
    return not file_name.endswith('.pkl')


# def load_runs(dirs, height=84, width=84, downsample=None, min_score=np.inf, **kwargs):
def load_runs(dirs, **kwargs):
    raw_runs = []
    for d in dirs:
        if os.path.isdir(d):
            for file_name in os.listdir(dirs):
                if is_not_pickle_file(file_name):
                    # do not process if the file is not pickle type
                    continue

                frame_path = os.path.join(d, file_name)
                with open(frame_path, 'rb') as f:
                    raw_run = pickle.load(f)
                    raw_runs.append(raw_run)
        else:
            zipped_file = tarfile.open(d, 'r:gz')

            for file_name in zipped_file.getnames():
                if is_not_pickle_file(file_name):
                    # do not process if the file is not pickle type
                    continue

                frame_path = zipped_file.getmember(file_name)
                f = zipped_file.extractfile(frame_path)
                raw_run = pickle.load(f)
                raw_runs.append(raw_run)

    return [preprocess_run(raw_run, **kwargs) for raw_run in raw_runs]
