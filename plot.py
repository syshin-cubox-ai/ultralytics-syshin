import pathlib

import cv2

from ultralytics import YOLO

source_path = pathlib.Path("C:/Users/synml/Desktop/1.mp4").resolve(strict=True)
dest_path = source_path.parent.joinpath(f"{source_path.stem}_result{source_path.suffix}").resolve()


def main():
    model = YOLO("yolov8x-pose.pt")

    vid_writer = None
    results = model.predict(source_path, stream=True)
    for result in results:
        if vid_writer is None:
            suffix, fourcc = (".mp4", "avc1")
            vid_writer = cv2.VideoWriter(
                filename=str(dest_path.with_suffix(suffix)),
                fourcc=cv2.VideoWriter.fourcc(*fourcc),
                fps=30,
                frameSize=(result.orig_shape[1], result.orig_shape[0]),
            )
        img = result.plot(conf=False, kpt_radius=3, kpt_line=True, labels=False, boxes=False)
        vid_writer.write(img)
    vid_writer.release()


if __name__ == "__main__":
    main()
