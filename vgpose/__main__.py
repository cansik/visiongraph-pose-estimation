import argparse

import visiongraph as vg
from visiongraph.input import add_input_step_choices

from vgpose.VGPoseGraph import VGPoseGraph


def main():
    args = parse_args()

    vg.setup_logging(args.loglevel)

    pipeline = VGPoseGraph(args.input(), args.pose_estimator())
    pipeline.configure(args)
    pipeline.open()


def parse_args():
    parser = argparse.ArgumentParser("vgpose", description="Example Pipeline")
    vg.add_logging_parameter(parser)

    VGPoseGraph.add_params(parser)

    input_group = parser.add_argument_group("input provider")
    add_input_step_choices(input_group)

    pose_group = parser.add_argument_group("pose estimator")
    vg.add_pose_estimation_step_choices(pose_group)

    return parser.parse_args()


if __name__ == "__main__":
    main()
