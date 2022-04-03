from argparse import ArgumentParser, Namespace

import cv2
import visiongraph as vg


class VGPoseGraph(vg.BaseGraph):

    def __init__(self, input: vg.BaseInput, pose_network: vg.PoseEstimator):
        super().__init__()
        self.input = input
        self.network = pose_network
        self.fps_tracer = vg.FPSTracer()

        self.performance_profiling = False

        self.add_nodes(self.input, self.network)

    def _process(self):
        ts, frame = self.input.read()

        if frame is None:
            return

        results = self.network.process(frame)

        for result in results:
            result.annotate(frame, min_score=0.1)

        self.fps_tracer.update()

        if not self.performance_profiling:
            cv2.putText(frame, "FPS: %.0f" % self.fps_tracer.smooth_fps,
                        (7, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2, cv2.LINE_AA)

            cv2.imshow("Pose Estimator", frame)
            if cv2.waitKey(15) & 0xFF == 27:
                self.close()
        else:
            print("\033[K", end='')
            print("FPS: %.0f" % self.fps_tracer.smooth_fps)

    @staticmethod
    def add_params(parser: ArgumentParser):
        super(VGPoseGraph, VGPoseGraph).add_params(parser)
        parser.add_argument("-p", "--performance", action="store_true", help="Enable performance profiling (no UI).")

    def configure(self, args: Namespace):
        super().configure(args)

        self.performance_profiling = args.performance


