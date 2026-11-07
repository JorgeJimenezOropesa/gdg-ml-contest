from pathlib import Path
import numpy as np
import onnxruntime as ort
from competition_api import ABSTAIN, Sample
from preprocess import inference_preprocess_sample

HERE = Path(__file__).parent
MODEL_PATH = HERE / "model.onnx"


class Model:
    def __init__(self):
        # Configure ONNX Runtime to strictly use 1 thread as required by the contest
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        # Load ONNX graph
        self.session = ort.InferenceSession(
            str(MODEL_PATH),
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name

    def predict(self, x: Sample) -> int:
        # Handle NaNs
        crops = inference_preprocess_sample(x.series, x.meta)

        # Reshape to batch dimension: (1, 4, 128) or (2, 4, 128)
        input_data = np.stack(crops, axis=0)

        # Inference
        logits = self.session.run(None, {self.input_name: input_data})[0]
        return int(np.argmax(logits.mean(axis=0)))
