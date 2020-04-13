import click
from loguru import logger

import swing3d.constants
import swing3d.utils
import swing3d.pose3d
import swing3d.keypoints


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
    keypoints, resolution = swing3d.keypoints.keypoints(
        input_video, output_path=None, small_model=small_kps_model, debug=debug
    )

    swing3d.pose3d.pose3d(keypoints, resolution, output_path=output_path)


if __name__ == "__main__":
    main()
