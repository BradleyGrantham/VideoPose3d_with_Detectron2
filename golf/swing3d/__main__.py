import click
from loguru import logger

import golf.swing3d.constants
import golf.swing3d.utils
import golf.swing3d.pose3d
import golf.swing3d.keypoints


@click.command()
@click.argument("input-video")
@click.option("--output-path")
@click.option(
    "--small-kps-model/--large-kps-model",
    default=True,
    help="Default is the small kps model.",
)
@click.option("--debug/--not-debug", default=False)
def main(input_video, output_path, small_kps_model, debug):
    keypoints, resolution = golf.swing3d.keypoints.keypoints(
        input_video, output_path=None, small_model=small_kps_model, debug=debug
    )

    golf.swing3d.pose3d.pose3d(keypoints, resolution, output_path=output_path)


if __name__ == "__main__":
    main()
