use burn::optim::AdamConfig;
use burn::backend::{WgpuBackend, wgpu::AutoGraphicsApi};
use burn::autodiff::ADBackendDecorator;

mod model;
mod data;
mod train;
mod infer;

fn main() {
    type MyBackend = WgpuBackend<AutoGraphicsApi, f32, i32>;
    type MyAutodiffBackend = ADBackendDecorator<MyBackend>;

    let device = burn::backend::wgpu::WgpuDevice::default();
    let artifact_dir = "/tmp/guide";
    train::train::<MyAutodiffBackend>(
        artifact_dir,
        train::TrainingConfig::new(model::ModelConfig::new(10, 512), AdamConfig::new()),
        device.clone(),
    );
}