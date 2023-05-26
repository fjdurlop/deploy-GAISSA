- [Steps to load a model into the API](#steps-to-load-a-model-into-the-api)
  - [Requirements](#requirements)
  - [FAQ](#faq)

# Steps to load a model into the API

## Requirements

## FAQ


- Trying to use pipeline with old or small CPU/GPU: Illegal instruction (core dumped)
>>> from transformers import pipeline
>>> import tensorflow
Illegal instruction (core dumped)

In brief, the error will be thrown if we’re running recent TensorFlow binaries on CPU(s) 
that do not support Advanced Vector Extensions (AVX), an instruction set that enables faster
computation especially for vector operations. Starting from TensorFlow 1.6, pre-built
TensorFlow binaries use AVX instructions. An excerpt from TensorFlow 1.6 release announcement: 
tf 1.6 - feb 18
transformers - 19
https://tech.amikelive.com/node-887/how-to-resolve-error-illegal-instruction-core-dumped-when-running-import-tensorflow-in-a-python-program/

flags           : fpu de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pse36 clflush mmx fxsr sse sse2 syscall nx lm rep_good nopl xtopology cpuid tsc_known_freq pni cx16 x2apic hypervisor lahf_lm cpuid_fault pti

CPU features
    windows
        check system information, then search {cpu model} CPU features
    linux
        more /proc/cpuinfo | grep flags
        
This repository is tested on Python 3.6+, Flax 0.3.2+, PyTorch 1.3.1+ and TensorFlow 2.3+.