# MVT inference using ONNX-Runtime and TensorRT as backends

## Please Note!
**Due to some conflicts in the inference script, we have created a separate branch for ONNX-Runtime and TensorRT-based evaluation. Hence, run the following commands on terminal to switch to the appropriate branch**
```
git clone https://github.com/goutamyg/MVT.git
git checkout --track origin/multi_framework_inference
```
---

## ONNX-Runtime Inference
**Introduction**：[ONNXRUNTIME](https://github.com/microsoft/onnxruntime) is an open-source library by Microsoft for network inference acceleration. The accelerated *MVT* tracker runs at 70*fps* :zap: on a 12th Gen Intel(R) Core-i9 CPU.

### Installation
For ONNX-Runtime-based inference on CPU, install
```
pip install onnx onnxruntime
```

### Pytorch to ONNX model
Download the onnx model from [here](https://drive.google.com/drive/folders/1RAdn3ZXI_G7pBj4NDbtQVFPkClVd1IBm)

***or***

Run ``` python tracking/pytorch2onnx.py ``` to generate the onnx file from the pretrained [pytorch model](https://drive.google.com/drive/folders/1RAdn3ZXI_G7pBj4NDbtQVFPkClVd1IBm)

### ONNX-Runtime inference
python tracking/test.py --tracker_name mobilevit_track --tracker_param mobilevit_256_128x1_got10k_ep100_cosine_annealing --dataset got10k_test --backend onnx

---

## TensorRT Conversion and Inference
**Introduction** [TensorRT](https://github.com/NVIDIA/TensorRT) is a high-performance deep learning inference SDK by NVIDIA:registered:. Using TensorRT as the backend, our *MVT* tracker runs at a speed of **~300***fps* :zap::zap: on an NVidia RTX 3090 GPU.

### Installation
```
pip install tensorrt
```

### Creating a TensorRT engine from Pytorch-based model (*via* ONNX)
[Since TensorRT engine should be built on a GPU of the same type as the one on which inference will be performed](https://blog.tensorflow.org/2021/01/leveraging-tensorflow-tensorrt-integration.html#:~:text=The%20TensorRT%20execution%20engine%20should,building%20process%20is%20GPU%20specific.), please run

Run ``` python tracking/pytorch2onnx.py ``` to generate the onnx file,
and ``` python tracking/onnx2trt.py ``` to generate TensorRT engine.

### TensorRT-based inference
python tracking/test.py --tracker_name mobilevit_track --tracker_param mobilevit_256_128x1_got10k_ep100_cosine_annealing --dataset got10k_test --backend tensorrt

---

## MVT v.s Other Trackers
| Tracker | Source | GOT10K-test (AUC)| Speed (CPU) | Speed (GPU) |
|---|---|---|---|---|
|**MVT**|BMVC'23|**0.633**|**~70***fps*|**~300***fps*|
|Ocean|ECCV'20|0.611|~10*fps*|~130*fps*|
|DiMP50|ICCV'19|0.611|~15*fps*|~60*fps*|

Our *MVT* tracker achieves better performance than DCF-based [DiMP50](https://github.com/visionml/pytracking) and Siamese-based [Ocean](https://github.com/researchmm/TracKit), while running at least ***4.5*** and ***2.3*** times faster on CPU and GPU :fire:, respectively.

 
