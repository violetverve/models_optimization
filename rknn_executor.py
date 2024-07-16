# The code is based on
# https://github.com/airockchip/rknn_model_zoo/blob/main/py_utils/rknn_executor.py
# with usage of RKNNLite instread of RKNN 
# replace the rknn_executor.py with this script

from rknnlite.api import RKNNLite

class RKNN_model_container():
    def __init__(self, model_path, target=None, device_id=None) -> None:
        rknn = RKNNLite()

        rknn.load_rknn(model_path)

        ret = rknn.init_runtime()
        
        if ret != 0:
            print('Init runtime environment failed')
            exit(ret)
        print('done')

        self.rknn = rknn 

    def run(self, inputs):
        if isinstance(inputs, list) or isinstance(inputs, tuple):
            pass
        else:
            inputs = [inputs]

        result = self.rknn.inference(inputs=inputs)
    
        return result


