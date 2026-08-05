from modules.deep_learning import load_model_bundle, predict_cnn
import pandas as pd
import numpy as np

print('loading bundle...')
bundle = load_model_bundle()
print('bundle ok:', bundle.get('ok'))

frame = pd.DataFrame([np.random.randint(0,256,784)])
print('running predict_cnn...')
res = predict_cnn(bundle, frame)
print('predicted:', res['predictions'][0])
print('confidence:', float(res['confidence'][0]))
print('done')
