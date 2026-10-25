Platform: Windows AMD64
Python: 3.13.16
GPU: NVIDIA GeForce RTX 4060 Laptop GPU

WARNING:root:Please consider to run pre-processing before quantization. Refer to example: https://github.com/microsoft/onnxruntime-inference-examples/blob/main/quantization/image_classification/cpu/ReadMe.md 
WARNING:root:Please consider to run pre-processing before quantization. Refer to example: https://github.com/microsoft/onnxruntime-inference-examples/blob/main/quantization/image_classification/cpu/ReadMe.md 
WARNING:root:Please consider pre-processing before quantization. See https://github.com/microsoft/onnxruntime-inference-examples/blob/main/quantization/image_classification/cpu/ReadMe.md 
  FP32 (Original)                    mean=0.022ms P50=0.015ms P95=0.042ms FPS=45730.8
  FP16                               mean=0.026ms P50=0.016ms P95=0.045ms FPS=38244.7
  INT8 Dynamic                       mean=0.066ms P50=0.028ms P95=0.121ms FPS=15230.2
  INT8 Static (random-norm calib.)   mean=0.079ms P50=0.042ms P95=0.130ms FPS=12694.7

Model                              Mean(ms)  P95(ms)   FPS       Action MSE   Verdict
--------------------------------------------------------------------------------------
FP32 (Original)                     0.022     0.042     45730.8   0.00000000   baseline
FP16                                0.026     0.045     38244.7   0.00000000   pass
INT8 Dynamic                        0.066     0.121     15230.2   0.00000273   pass
INT8 Static (random-norm calib.)    0.079     0.130     12694.7   0.00000478   pass
