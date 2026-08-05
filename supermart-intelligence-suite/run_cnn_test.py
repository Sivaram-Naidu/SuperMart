import json
import traceback
from modules.deep_learning import load_model_bundle, predict_cnn
import pandas as pd
import numpy as np

result = {}
try:
    bundle = load_model_bundle()
    result["bundle_ok"] = bool(bundle.get("ok"))
    if bundle.get("ok"):
        frame = pd.DataFrame([np.random.randint(0, 256, 784)])
        res = predict_cnn(bundle, frame)
        result.update({
            "predicted": res["predictions"][0],
            "confidence": float(res["confidence"][0]),
            "probabilities_shape": list(res["probabilities"].shape),
        })
except Exception as exc:
    result["error"] = str(exc)
    result["traceback"] = traceback.format_exc()

with open('supermart-intelligence-suite/cnn_test_result.json', 'w', encoding='utf-8') as fh:
    json.dump(result, fh)

print('wrote result')
