from argparse import ArgumentParser, Namespace

import cv2
import visiongraph as vg
from rich.console import Console
from rich.live import Live
from rich.padding import Padding
from rich.panel import Panel
from rich.text import Text


class VGPoseGraph(vg.BaseGraph):

    def __init__(self, input: vg.BaseInput, pose_network: vg.PoseEstimator):
        super().__init__(handle_signals=True)
        self.input = input
        self.network = pose_network
        self.fps_tracer = vg.FPSTracer()

        self.add_nodes(self.input, self.network)

        self.performance_profiling = False

        self.input_watch = vg.ProfileWatch()
        self.process_watch = vg.ProfileWatch()
        self.total_watch = vg.ProfileWatch()

        self.console = Console()
        self.console.print("VisionGraph Pose Estimation", style="bold green")

        self.live_field = Live(console=self.console, screen=False)
        self.panel = Panel("",
                           title="VisionGraph Pose Estimation",
                           title_align="left",
                           padding=1)

    def _init(self):
        with self.console.status("Starting pipeline..."):
            if isinstance(self.network, vg.OpenVinoPoseEstimator):
                self.network.device = vg.get_inference_engine_device()

            if self.performance_profiling and isinstance(self.input, vg.VideoCaptureInput):
                self.input.fps_lock = False

            super()._init()
        self.live_field.start()

    def _process(self):
        self.total_watch.start()
        self.input_watch.start()
        ts, frame = self.input.read()
        self.input_watch.stop()

        if frame is None:
            return

        self.process_watch.start()
        results = self.network.process(frame)
        self.process_watch.stop()

        self.fps_tracer.update()

        if not self.performance_profiling:
            for result in results:
                result.annotate(frame, min_score=0.1, show_bounding_box=False)

            cv2.putText(frame, "FPS: %.0f" % self.fps_tracer.smooth_fps,
                        (7, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (128, 128, 128), 2, cv2.LINE_AA)

            cv2.imshow("Pose Estimator", frame)
            if cv2.waitKey(15) & 0xFF == 27:
                self.close()

        self.total_watch.stop()

        h, w = frame.shape[:2]

        text = Text(justify="center")
        text.append("Input")
        text.append("(ms)", style="italic")
        text.append(": ")
        text.append(f"{self.input_watch.average():.1f}", style="bold cyan")
        text.append("    ")
        text.append("Estimation")
        text.append("(ms)", style="italic")
        text.append(": ")
        text.append(f"{self.process_watch.average():.1f}", style="bold yellow")
        text.append("    ")
        text.append("Total")
        text.append("(ms)", style="italic")
        text.append(": ")
        text.append(f"{self.total_watch.average():.1f}", style="bold magenta")
        text.append("    ")
        text.append("FPS: ")
        text.append(f"{self.fps_tracer.smooth_fps:.1f}", style="bold magenta")

        self.panel.renderable = text
        self.panel.subtitle = f"{type(self.network).__name__} ({h} x {w})"
        self.live_field.update(Padding(self.panel, 1))

    def _release(self):
        self.live_field.stop()
        with self.console.status("Stopping pipeline..."):
            super()._release()

    @staticmethod
    def add_params(parser: ArgumentParser):
        super(VGPoseGraph, VGPoseGraph).add_params(parser)
        parser.add_argument("-p", "--performance", action="store_true", help="Enable performance profiling (no UI).")

    def configure(self, args: Namespace):
        super().configure(args)

        self.performance_profiling = args.performance
