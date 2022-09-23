# Vehicle-Detection-YOLOv3
+ The Project aims to detect vehicles in a live or pre-rendered video feed on the Single Board Computers (SBCs) like Raspberry Pi or the Banana Pi, where GPU acceleration is not available for concurrent floating-point operations.
+ Deep Learning traditionally uses 32-bit floating-point operations, which run concurrently on Nvidia’s CUDA-compatible stream units. This is not possible in Embedded Computers as they typically do not possess GPU acceleration and the ARM Instruction Set is not usually good at handling long-bit operations.
+ The existing neural network structure of the Yolov3-tiny (You Only Look Once) framework was used to support the CPU-based approach taken for this problem. 
+ The implemented neural network was shallow and optimized for downsampling. The Delayed-down sampling allowed external networks to extract as many features from the image as possible.
+ The proposed solution slightly decreased accuracy but significantly improved performance for cheaper hardware.
