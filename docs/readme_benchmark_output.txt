Platform: Windows AMD64
Python: 3.13.16
GPU: NVIDIA GeForce RTX 4060 Laptop GPU

WARNING:root:Please consider to run pre-processing before quantization. Refer to example: https://github.com/microsoft/onnxruntime-inference-examples/blob/main/quantization/image_classification/cpu/ReadMe.md 
WARNING:root:Please consider to run pre-processing before quantization. Refer to example: https://github.com/microsoft/onnxruntime-inference-examples/blob/main/quantization/image_classification/cpu/ReadMe.md 
WARNING:root:Please consider pre-processing before quantization. See https://github.com/microsoft/onnxruntime-inference-examples/blob/main/quantization/image_classification/cpu/ReadMe.md 
  FP32 (Original)                    mean=0.019ms P50=0.014ms P95=0.037ms FPS=51486.1
  FP16                               mean=0.047ms P50=0.017ms P95=0.085ms FPS=21465.2
  INT8 Dynamic                       mean=0.064ms P50=0.039ms P95=0.130ms FPS=15676.3
  INT8 Static (random-norm calib.)   mean=0.044ms P50=0.031ms P95=0.109ms FPS=22781.1

Model                              Mean(ms)  P95(ms)   FPS       Action MSE   Verdict
--------------------------------------------------------------------------------------
FP32 (Original)                     0.019     0.037     51486.1   0.00000000   baseline
FP16                                0.047     0.085     21465.2   0.00000000   pass
INT8 Dynamic                        0.064     0.130     15676.3   0.00000250   pass
INT8 Static (random-norm calib.)    0.044     0.109     22781.1   0.00000548   pass
