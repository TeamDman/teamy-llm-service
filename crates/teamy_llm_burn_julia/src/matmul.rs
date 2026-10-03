//! Explicit whole-matrix FP32 execution without accelerated TF32 selection.
use burn::backend::{Cuda, NdArray};
use burn::tensor::{DType, Tensor, TensorPrimitive, backend::Backend};
use burn_cubecl::kernel::matmul::init_matmul_output;
use cubek::matmul::{
    definition::{MatmulElems, MatmulGlobalElems},
    launch::{Strategy, launch_ref},
};
use cubek::std::InputBinding;

/// Julia's qualified backends select their own full-precision matmul path.
pub(crate) trait JuliaBackend: Backend {
    fn fp32_matmul(lhs: Tensor<Self, 4>, rhs: Tensor<Self, 4>) -> Tensor<Self, 4>;
}

impl JuliaBackend for NdArray<f32> {
    fn fp32_matmul(lhs: Tensor<Self, 4>, rhs: Tensor<Self, 4>) -> Tensor<Self, 4> {
        lhs.matmul(rhs)
    }
}

impl JuliaBackend for Cuda<f32, i32> {
    fn fp32_matmul(lhs: Tensor<Self, 4>, rhs: Tensor<Self, 4>) -> Tensor<Self, 4> {
        let lhs = lhs.into_primitive().tensor();
        let rhs = rhs.into_primitive().tensor();
        assert_eq!(lhs.dtype, DType::F32, "Julia matmul requires FP32 inputs");
        assert_eq!(rhs.dtype, DType::F32, "Julia matmul requires FP32 inputs");
        let output = init_matmul_output(&lhs, &rhs, DType::F32);
        let mut dtypes = MatmulElems::from_globals(&MatmulGlobalElems {
            lhs: DType::F32.into(),
            rhs: DType::F32.into(),
            out: DType::F32.into(),
        });
        // SimpleUnit uses the Register tile implementation, retaining FP32
        // globals, stages and registers. Auto selects tensor-core strategies
        // that can convert F32 operands to TF32; it is deliberately excluded.
        // Bindings preserve actual shape/strides, including transposed weights
        // and broadcast batch/head dimensions. This also bypasses autotuning.
        launch_ref(
            &Strategy::SimpleUnit(Default::default()),
            &output.client,
            InputBinding::new(lhs.binding(), DType::F32.into()),
            InputBinding::new(rhs.binding(), DType::F32.into()),
            output.clone().binding(),
            &mut dtypes,
        )
        .expect("Julia FP32 SimpleUnit matmul launch failed");
        Tensor::from_primitive(TensorPrimitive::Float(output))
    }
}
