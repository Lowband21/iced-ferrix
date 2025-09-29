use crate::core::{self, Size};
use crate::image::atlas::{self, Atlas};

use std::sync::Arc;

#[derive(Debug)]
pub struct Cache {
    atlas: Atlas,
    #[cfg(feature = "image")]
    raster: crate::image::raster::Cache,
    #[cfg(feature = "svg")]
    vector: crate::image::vector::Cache,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AtlasRegion {
    pub uv_min: [f32; 2],
    pub uv_max: [f32; 2],
    pub layer: u32,
}

impl Cache {
    pub fn new(
        device: &wgpu::Device,
        backend: wgpu::Backend,
        layout: Arc<wgpu::BindGroupLayout>,
    ) -> Self {
        Self {
            atlas: Atlas::new(device, backend, layout),
            #[cfg(feature = "image")]
            raster: crate::image::raster::Cache::default(),
            #[cfg(feature = "svg")]
            vector: crate::image::vector::Cache::default(),
        }
    }

    pub fn bind_group(&self) -> &wgpu::BindGroup {
        self.atlas.bind_group()
    }

    pub fn layer_count(&self) -> usize {
        self.atlas.layer_count()
    }

    pub fn texture_layout(&self) -> Arc<wgpu::BindGroupLayout> {
        self.atlas.bind_group_layout()
    }

    #[cfg(feature = "image")]
    pub fn measure_image(&mut self, handle: &core::image::Handle) -> Size<u32> {
        self.raster.load(handle).dimensions()
    }

    #[cfg(feature = "svg")]
    pub fn measure_svg(&mut self, handle: &core::svg::Handle) -> Size<u32> {
        self.vector.load(handle).viewport_dimensions()
    }

    #[cfg(feature = "image")]
    pub fn upload_raster(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        handle: &core::image::Handle,
    ) -> Option<&atlas::Entry> {
        self.raster.upload(device, encoder, handle, &mut self.atlas)
    }

    #[cfg(feature = "image")]
    pub fn ensure_raster_region(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        handle: &core::image::Handle,
    ) -> Option<AtlasRegion> {
        let entry = self.upload_raster(device, encoder, handle)?;
        Some(Self::region_from_entry(entry))
    }

    #[cfg(feature = "image")]
    pub fn cached_raster_region(
        &mut self,
        handle: &core::image::Handle,
    ) -> Option<AtlasRegion> {
        // Record cache hits so atlas entries persist across trims.
        self.raster.atlas_entry(handle).map(Self::region_from_entry)
    }

    #[cfg(feature = "svg")]
    pub fn upload_vector(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        handle: &core::svg::Handle,
        color: Option<core::Color>,
        size: [f32; 2],
        scale: f32,
    ) -> Option<&atlas::Entry> {
        self.vector.upload(
            device,
            encoder,
            handle,
            color,
            size,
            scale,
            &mut self.atlas,
        )
    }

    pub fn trim(&mut self) {
        #[cfg(feature = "image")]
        self.raster.trim(&mut self.atlas);

        #[cfg(feature = "svg")]
        self.vector.trim(&mut self.atlas);
    }
}

impl Cache {
    fn region_from_entry(entry: &atlas::Entry) -> AtlasRegion {
        match entry {
            atlas::Entry::Contiguous(allocation) => {
                let (x, y) = allocation.position();
                let size = allocation.size();
                let layer = allocation.layer() as u32;
                let atlas_size = atlas::SIZE as f32;

                AtlasRegion {
                    uv_min: [x as f32 / atlas_size, y as f32 / atlas_size],
                    uv_max: [
                        (x + size.width) as f32 / atlas_size,
                        (y + size.height) as f32 / atlas_size,
                    ],
                    layer,
                }
            }
            atlas::Entry::Fragmented { size, fragments } => {
                let atlas_size = atlas::SIZE as f32;

                if let Some(first) = fragments.first() {
                    let (x, y) = first.position;
                    let layer = first.allocation.layer() as u32;

                    AtlasRegion {
                        uv_min: [x as f32 / atlas_size, y as f32 / atlas_size],
                        uv_max: [
                            (x + size.width) as f32 / atlas_size,
                            (y + size.height) as f32 / atlas_size,
                        ],
                        layer,
                    }
                } else {
                    AtlasRegion {
                        uv_min: [0.0, 0.0],
                        uv_max: [0.0, 0.0],
                        layer: 0,
                    }
                }
            }
        }
    }
}

#[cfg(all(test, feature = "image"))]
mod tests {
    use super::*;
    use crate::core::image::Handle;

    fn create_device() -> (wgpu::Device, wgpu::Queue, wgpu::Backend) {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(
            &wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::LowPower,
                force_fallback_adapter: false,
                compatible_surface: None,
            },
        ))
        .expect("request adapter");

        let backend = adapter.get_info().backend;

        let descriptor = wgpu::DeviceDescriptor {
            label: Some("batched-image-test-device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            memory_hints: wgpu::MemoryHints::default(),
            ..Default::default()
        };

        let (device, queue) =
            pollster::block_on(adapter.request_device(&descriptor))
                .expect("request device");

        (device, queue, backend)
    }

    fn create_layout(device: &wgpu::Device) -> Arc<wgpu::BindGroupLayout> {
        Arc::new(device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: Some("test atlas layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float {
                            filterable: true,
                        },
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                        multisampled: false,
                    },
                    count: None,
                }],
            },
        ))
    }

    fn solid_handle(seed: u8) -> Handle {
        const WIDTH: u32 = 4;
        const HEIGHT: u32 = 4;

        let mut pixels = Vec::with_capacity((WIDTH * HEIGHT * 4) as usize);

        for _ in 0..(WIDTH * HEIGHT) {
            pixels.extend_from_slice(&[seed, 255 - seed, seed / 2, 255]);
        }

        Handle::from_rgba(WIDTH, HEIGHT, pixels)
    }

    #[test]
    fn cached_regions_survive_trim_after_hit() {
        let (device, queue, backend) = create_device();
        let layout = create_layout(&device);
        let mut cache = Cache::new(&device, backend, layout);

        let handle = solid_handle(64);
        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("batched-image-test-encoder"),
            });

        assert!(
            cache
                .ensure_raster_region(&device, &mut encoder, &handle)
                .is_some()
        );

        let submission = queue.submit([encoder.finish()]);
        let _ = device.poll(wgpu::PollType::WaitForSubmissionIndex(submission));

        // First frame trim: retains the allocation and resets cache bookkeeping.
        cache.trim();

        // Simulate a second frame: upload a new handle while touching the old
        // one to register a cache hit before the next trim.
        let new_handle = solid_handle(192);
        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("batched-image-test-encoder-second"),
            });

        assert!(
            cache
                .ensure_raster_region(&device, &mut encoder, &new_handle)
                .is_some()
        );

        let submission = queue.submit([encoder.finish()]);
        let _ = device.poll(wgpu::PollType::WaitForSubmissionIndex(submission));

        assert!(cache.cached_raster_region(&handle).is_some());
        assert!(cache.cached_raster_region(&new_handle).is_some());

        cache.trim();

        assert!(cache.cached_raster_region(&handle).is_some());
        assert!(cache.cached_raster_region(&new_handle).is_some());
    }
}
