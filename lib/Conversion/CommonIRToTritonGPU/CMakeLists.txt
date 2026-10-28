add_triton_library(CommonIRToTritonGPU
  SetLayout.cpp
  TensorTiles.cpp

  LINK_LIBS PUBLIC
  MLIRTransforms
  CommonIRIR
  CommonIRTransforms
  TritonGPUIR
)
