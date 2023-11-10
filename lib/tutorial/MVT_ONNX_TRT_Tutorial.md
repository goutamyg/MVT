# MVT inference using ONNX-Runtime and TensorRT as backends

## Please Note!
**Due to some conflicts in the inference code, we have created a separate branch for ONNX-Runtime and TensorRT-based evaluation. Hence, run the following commands on terminal to switch to the appropriate branch**
```
git clone https://github.com/goutamyg/MVT.git
git checkout --track origin/multi_framework_inference
```

## ONNX-Runtime Inference
**Introduction**：[ONNXRUNTIME](https://github.com/microsoft/onnxruntime) is an open-source library by Microsoft for network inference acceleration. The accelerated MVT runs at 70 FPS on a 12th Gen Intel(R) Core-i9 CPU! 

### Installation
For ONNX-Runtime-based inference on CPU, install
```
pip install onnx onnxruntime
```

### Pytorch to ONNX model
Download the onnx model from [here](https://drive.google.com/drive/folders/1RAdn3ZXI_G7pBj4NDbtQVFPkClVd1IBm)

***or***

Run ``` python tracking/pytorch2onnx.py ``` to generate the onnx file from the pretrained [pytorch model](https://drive.google.com/drive/folders/1RAdn3ZXI_G7pBj4NDbtQVFPkClVd1IBm)

### MVT v.s Other Trackers on CPU
| Tracker | GOT10K-test (AUC)| Speed (FPS) |
|---|---|---|
|**MVT**|**0.633**|**~70**|
|DiMP50|0.611|~15|
|Ocean|0.611|~10|

MVT achieves better performance than DCF-based DiMP50 and Siamese Network-based Ocean, while running at 70 *fps*:zap: on CPU.

# TensorRT Conversion and Inference
