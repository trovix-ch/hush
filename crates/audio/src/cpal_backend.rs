use std::sync::Arc;
use std::sync::atomic::Ordering;

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{ErrorKind, FromSample, SampleFormat, SizedSample, StreamConfig};
use rtrb::Producer;
use wl_core::recorder::RecorderError;

use crate::capture::{CallbackShared, OpenedStream, StreamOpener, downmix_into, note_dropped};

#[derive(Debug, Clone)]
pub struct InputDeviceInfo {
    pub name: String,
    /// Preferable to the name in config because two identical USB mics share a name.
    pub id: String,
    pub is_default: bool,
    pub default_format: Option<String>,
}

pub fn list_input_devices() -> Result<Vec<InputDeviceInfo>, RecorderError> {
    let host = cpal::default_host();
    let default_id = host.default_input_device().and_then(|d| d.id().ok());
    let devices = host
        .input_devices()
        .map_err(|e| RecorderError::Device(e.to_string()))?;
    let mut out = Vec::new();
    for d in devices {
        let id = d.id().ok();
        let name = d
            .description()
            .map(|x| x.name().to_owned())
            .unwrap_or_else(|_| "<unnamed>".into());
        let default_format = d.default_input_config().ok().map(|c| {
            format!(
                "{} Hz, {} ch, {}",
                c.sample_rate(),
                c.channels(),
                c.sample_format()
            )
        });
        out.push(InputDeviceInfo {
            name,
            is_default: id.is_some() && id == default_id,
            id: id.map(|i| i.to_string()).unwrap_or_default(),
            default_format,
        });
    }
    Ok(out)
}

#[derive(Debug, Default)]
pub struct CpalOpener;

fn find_device(host: &cpal::Host, wanted: Option<&str>) -> Result<cpal::Device, RecorderError> {
    let Some(wanted) = wanted else {
        return host
            .default_input_device()
            .ok_or(RecorderError::NoInputDevice);
    };
    let devices: Vec<cpal::Device> = host
        .input_devices()
        .map_err(|e| RecorderError::Device(e.to_string()))?
        .collect();
    if devices.is_empty() {
        return Err(RecorderError::NoInputDevice);
    }
    let name_of = |d: &cpal::Device| d.description().map(|x| x.name().to_owned()).ok();
    let id_of = |d: &cpal::Device| d.id().ok().map(|i| i.to_string());
    let lower = wanted.to_lowercase();
    // Substring as a last resort because Windows decorates names ("Microphone (2- USB
    // Audio)") when it re-enumerates.
    devices
        .iter()
        .find(|d| id_of(d).as_deref() == Some(wanted))
        .or_else(|| {
            devices
                .iter()
                .find(|d| name_of(d).as_deref() == Some(wanted))
        })
        .or_else(|| {
            devices.iter().find(|d| {
                name_of(d)
                    .map(|n| n.to_lowercase().contains(&lower))
                    .unwrap_or(false)
            })
        })
        .cloned()
        .ok_or_else(|| RecorderError::Device(format!("input device not found: {wanted}")))
}

fn map_err(e: cpal::Error) -> RecorderError {
    match e.kind() {
        ErrorKind::DeviceNotAvailable | ErrorKind::PermissionDenied | ErrorKind::DeviceBusy => {
            RecorderError::Device(e.to_string())
        }
        _ => RecorderError::Stream(e.to_string()),
    }
}

fn build<T>(
    device: &cpal::Device,
    config: StreamConfig,
    mut producer: Producer<f32>,
    shared: Arc<CallbackShared>,
) -> Result<cpal::Stream, RecorderError>
where
    T: SizedSample,
    f32: FromSample<T>,
{
    let channels = config.channels as usize;
    let data_shared = shared.clone();
    let err_shared = shared;
    device
        .build_input_stream::<T, _, _>(
            config,
            move |data: &[T], _| {
                let n = downmix_into(data, channels, &mut producer, |s: T| s.to_sample::<f32>());
                note_dropped(&data_shared, n);
            },
            move |err: cpal::Error| {
                // WASAPI never rebinds an open client, so a default-device change arrives
                // as StreamInvalidated and treating it as recoverable would record silence.
                if !matches!(err.kind(), ErrorKind::Xrun | ErrorKind::RealtimeDenied) {
                    err_shared.lost.store(true, Ordering::Release);
                }
            },
            None,
        )
        .map_err(map_err)
}

/// Held only to keep the stream alive; cpal's `Stream` is `Send` on every backend.
struct Holder(#[allow(dead_code)] cpal::Stream);

impl StreamOpener for CpalOpener {
    fn open(
        &mut self,
        device: Option<&str>,
        producer: Producer<f32>,
        shared: Arc<CallbackShared>,
    ) -> Result<OpenedStream, RecorderError> {
        let host = cpal::default_host();
        let device = find_device(&host, device)?;
        let device_name = device
            .description()
            .map(|d| d.name().to_owned())
            .unwrap_or_default();
        // Asking WASAPI shared mode for 16 kHz either fails or engages its resampler,
        // which is worse than ours.
        let supported = device.default_input_config().map_err(map_err)?;
        let config: StreamConfig = supported.into();
        let fmt = supported.sample_format();
        let stream = match fmt {
            SampleFormat::F32 => build::<f32>(&device, config, producer, shared),
            SampleFormat::F64 => build::<f64>(&device, config, producer, shared),
            SampleFormat::I16 => build::<i16>(&device, config, producer, shared),
            SampleFormat::I24 => build::<cpal::I24>(&device, config, producer, shared),
            SampleFormat::I32 => build::<i32>(&device, config, producer, shared),
            SampleFormat::U16 => build::<u16>(&device, config, producer, shared),
            SampleFormat::I8 => build::<i8>(&device, config, producer, shared),
            SampleFormat::U8 => build::<u8>(&device, config, producer, shared),
            other => Err(RecorderError::Device(format!(
                "unsupported sample format {other}"
            ))),
        }?;
        stream.play().map_err(map_err)?;
        tracing::debug!(
            device = %device_name,
            rate = config.sample_rate,
            channels = config.channels,
            format = %fmt,
            "capture stream playing"
        );
        Ok(OpenedStream {
            sample_rate: config.sample_rate,
            channels: config.channels,
            device_name,
            handle: Box::new(Holder(stream)),
        })
    }
}
