add_triton_library(CommonIRTransforms
  InferLayout.cpp
  BufferLayoutPropagation.cpp
  LayoutHints.cpp
  TensorLayoutPropagation.cpp

  DEPENDS
  CommonIRTransformsPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRArithDialect
  MLIRSCFDialect
  CommonIRIR
)

target_include_directories(CommonIRTransforms PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../../../include>
)
