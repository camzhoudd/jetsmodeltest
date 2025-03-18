import coremltools as ct
from coremltools.models.neural_network import quantization_utils

# load full precision model
model_fp32 = ct.models.MLModel('TagYOLOModel.mlmodel')

model_fp16 = quantization_utils.quantize_weights(model_fp32, nbits=16)

model_fp16.save("TagYOLOModel_16.mlmodel")
