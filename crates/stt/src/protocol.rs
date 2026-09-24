//! The pipe protocol between the app and `hush-stt-worker`.
//!
//! One frame: 4 magic bytes, the header length and the blob length as little-endian
//! `u32`, a JSON header, then the blob. Audio travels in the blob as raw little-endian
//! `f32`, so 30 s of speech is 1.9 MB on the pipe rather than 2.6 MB of base64 or a
//! JSON number per sample.

use std::io::{self, Read, Write};
use std::time::Duration;

use hush_core::UtteranceId;
use hush_core::stt::{Backend, Caps, EngineInfo, Segment, SttError, Transcript};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

pub const MAGIC: [u8; 4] = *b"HSW1";
/// A length past these is a corrupted stream, not a request: stray bytes read as a length
/// would otherwise allocate gigabytes.
const MAX_HEADER: usize = 1 << 20;
const MAX_BLOB: usize = 64 << 20;

/// `id` is echoed in the reply; `Cancel` carries the id of the request it cancels.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestFrame {
    pub id: u64,
    pub request: Request,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Request {
    /// `gpu` is the PCI bus id the caller resolved; a Vulkan index is never sent because
    /// the order differs between processes and sessions. `None` on Vulkan applies the
    /// auto policy in the worker.
    Load {
        gpu: Option<String>,
    },
    WarmUp,
    Nudge,
    /// The PCM is the frame's blob.
    Transcribe {
        utterance: u64,
        language: Option<String>,
        prompt: Option<String>,
        hotwords: Vec<String>,
        /// Time left on the caller's deadline when the request was sent.
        deadline_ms: Option<u64>,
    },
    Cancel,
    Info,
    Shutdown,
    /// Test hook: abort the process this long into the next transcription.
    Crash {
        after_ms: u64,
    },
}

/// Heartbeats carry id 0, which no request uses.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Reply {
    pub id: u64,
    pub body: ReplyBody,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", content = "data", rename_all = "snake_case")]
pub enum ReplyBody {
    Loaded { info: WireInfo, load_us: u64 },
    Done { elapsed_us: u64 },
    Transcript(WireTranscript),
    Info(WireInfo),
    Error(WireError),
    Heartbeat,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WireInfo {
    pub id: String,
    pub backend: String,
    pub device: Option<String>,
    pub prompt: bool,
    pub hotwords: bool,
    pub word_timestamps: bool,
    pub punctuation: bool,
    pub cancel: bool,
    pub languages: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WireSegment {
    pub text: String,
    pub start_us: u64,
    pub end_us: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WireTranscript {
    pub utterance: u64,
    pub text: String,
    pub segments: Vec<WireSegment>,
    pub words: Vec<WireSegment>,
    pub language: Option<String>,
    pub inference_us: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ErrorKind {
    Load,
    Backend,
    Inference,
    EmptyAudio,
    Cancelled,
    Deadline,
    BackendDied,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WireError {
    pub kind: ErrorKind,
    pub message: String,
}

pub fn backend_name(b: Backend) -> &'static str {
    match b {
        Backend::Cpu => "cpu",
        Backend::DirectMl => "directml",
        Backend::Cuda => "cuda",
        Backend::Vulkan => "vulkan",
    }
}

pub fn parse_backend(s: &str) -> Option<Backend> {
    match s.to_ascii_lowercase().as_str() {
        "cpu" => Some(Backend::Cpu),
        "directml" => Some(Backend::DirectMl),
        "cuda" => Some(Backend::Cuda),
        "vulkan" => Some(Backend::Vulkan),
        _ => None,
    }
}

fn us(d: Duration) -> u64 {
    u64::try_from(d.as_micros()).unwrap_or(u64::MAX)
}

impl From<&EngineInfo> for WireInfo {
    fn from(i: &EngineInfo) -> Self {
        Self {
            id: i.id.clone(),
            backend: backend_name(i.backend).into(),
            device: i.device.clone(),
            prompt: i.caps.prompt,
            hotwords: i.caps.hotwords,
            word_timestamps: i.caps.word_timestamps,
            punctuation: i.caps.punctuation,
            cancel: i.caps.cancel,
            languages: i.languages.clone(),
        }
    }
}

impl WireInfo {
    pub fn into_info(self) -> Result<EngineInfo, SttError> {
        let backend = parse_backend(&self.backend).ok_or_else(|| {
            SttError::Backend(format!(
                "worker reported unknown backend `{}`",
                self.backend
            ))
        })?;
        Ok(EngineInfo {
            id: self.id,
            backend,
            device: self.device,
            caps: Caps {
                prompt: self.prompt,
                hotwords: self.hotwords,
                word_timestamps: self.word_timestamps,
                punctuation: self.punctuation,
                cancel: self.cancel,
            },
            languages: self.languages,
        })
    }
}

impl From<&Segment> for WireSegment {
    fn from(s: &Segment) -> Self {
        Self {
            text: s.text.clone(),
            start_us: us(s.start),
            end_us: us(s.end),
        }
    }
}

impl From<WireSegment> for Segment {
    fn from(s: WireSegment) -> Self {
        Self {
            text: s.text,
            start: Duration::from_micros(s.start_us),
            end: Duration::from_micros(s.end_us),
        }
    }
}

impl From<&Transcript> for WireTranscript {
    fn from(t: &Transcript) -> Self {
        Self {
            utterance: t.utterance.0,
            text: t.text.clone(),
            segments: t.segments.iter().map(Into::into).collect(),
            words: t.words.iter().map(Into::into).collect(),
            language: t.language.clone(),
            inference_us: us(t.inference_time),
        }
    }
}

impl From<WireTranscript> for Transcript {
    fn from(t: WireTranscript) -> Self {
        Self {
            utterance: UtteranceId(t.utterance),
            text: t.text,
            segments: t.segments.into_iter().map(Into::into).collect(),
            words: t.words.into_iter().map(Into::into).collect(),
            language: t.language,
            inference_time: Duration::from_micros(t.inference_us),
        }
    }
}

impl From<&SttError> for WireError {
    fn from(e: &SttError) -> Self {
        let (kind, message) = match e {
            SttError::Load(m) => (ErrorKind::Load, m.clone()),
            SttError::Backend(m) => (ErrorKind::Backend, m.clone()),
            SttError::Inference(m) => (ErrorKind::Inference, m.clone()),
            SttError::EmptyAudio => (ErrorKind::EmptyAudio, String::new()),
            SttError::Cancelled => (ErrorKind::Cancelled, String::new()),
            SttError::Deadline => (ErrorKind::Deadline, String::new()),
            SttError::BackendDied(m) => (ErrorKind::BackendDied, m.clone()),
        };
        Self { kind, message }
    }
}

impl From<WireError> for SttError {
    fn from(e: WireError) -> Self {
        match e.kind {
            ErrorKind::Load => Self::Load(e.message),
            ErrorKind::Backend => Self::Backend(e.message),
            ErrorKind::Inference => Self::Inference(e.message),
            ErrorKind::EmptyAudio => Self::EmptyAudio,
            ErrorKind::Cancelled => Self::Cancelled,
            ErrorKind::Deadline => Self::Deadline,
            ErrorKind::BackendDied => Self::BackendDied(e.message),
        }
    }
}

pub fn pcm_to_bytes(pcm: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(pcm.len() * 4);
    for s in pcm {
        out.extend_from_slice(&s.to_le_bytes());
    }
    out
}

pub fn bytes_to_pcm(bytes: &[u8]) -> io::Result<Vec<f32>> {
    if !bytes.len().is_multiple_of(4) {
        return Err(invalid(format!(
            "PCM blob of {} bytes is not whole f32 samples",
            bytes.len()
        )));
    }
    let (chunks, _) = bytes.as_chunks::<4>();
    Ok(chunks.iter().map(|c| f32::from_le_bytes(*c)).collect())
}

fn invalid(msg: String) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg)
}

/// Written in one call so a frame is never interleaved with another thread's.
pub fn write_frame<W: Write, T: Serialize>(w: &mut W, header: &T, blob: &[u8]) -> io::Result<()> {
    let json = serde_json::to_vec(header).map_err(|e| invalid(e.to_string()))?;
    if json.len() > MAX_HEADER || blob.len() > MAX_BLOB {
        return Err(invalid(format!(
            "frame too large: header {} bytes, blob {} bytes",
            json.len(),
            blob.len()
        )));
    }
    let mut prefix = [0u8; 12];
    prefix[..4].copy_from_slice(&MAGIC);
    prefix[4..8].copy_from_slice(&(json.len() as u32).to_le_bytes());
    prefix[8..].copy_from_slice(&(blob.len() as u32).to_le_bytes());
    w.write_all(&prefix)?;
    w.write_all(&json)?;
    w.write_all(blob)?;
    w.flush()
}

/// `Ok(None)` is a clean end of stream between frames.
pub fn read_frame<R: Read, T: DeserializeOwned>(r: &mut R) -> io::Result<Option<(T, Vec<u8>)>> {
    let mut prefix = [0u8; 12];
    let mut got = 0;
    while got < prefix.len() {
        match r.read(&mut prefix[got..])? {
            0 if got == 0 => return Ok(None),
            0 => return Err(io::ErrorKind::UnexpectedEof.into()),
            n => got += n,
        }
    }
    if prefix[..4] != MAGIC {
        return Err(invalid(format!(
            "bad frame magic {:?}; something else wrote to the pipe",
            String::from_utf8_lossy(&prefix[..4])
        )));
    }
    let len = |b: &[u8]| u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as usize;
    let (header_len, blob_len) = (len(&prefix[4..8]), len(&prefix[8..]));
    if header_len > MAX_HEADER || blob_len > MAX_BLOB {
        return Err(invalid(format!(
            "frame too large: header {header_len} bytes, blob {blob_len} bytes"
        )));
    }
    let mut header = vec![0u8; header_len];
    r.read_exact(&mut header)?;
    let mut blob = vec![0u8; blob_len];
    r.read_exact(&mut blob)?;
    let header = serde_json::from_slice(&header).map_err(|e| invalid(e.to_string()))?;
    Ok(Some((header, blob)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frames_round_trip_with_audio() {
        let pcm: Vec<f32> = (0..1000).map(|i| (i as f32 * 0.01).sin()).collect();
        let mut buf = Vec::new();
        let req = RequestFrame {
            id: 7,
            request: Request::Transcribe {
                utterance: 3,
                language: Some("de".into()),
                prompt: None,
                hotwords: vec!["gRPC".into()],
                deadline_ms: Some(900),
            },
        };
        write_frame(&mut buf, &req, &pcm_to_bytes(&pcm)).unwrap();
        write_frame(
            &mut buf,
            &RequestFrame {
                id: 7,
                request: Request::Cancel,
            },
            &[],
        )
        .unwrap();
        let mut r = buf.as_slice();
        let (got, blob): (RequestFrame, _) = read_frame(&mut r).unwrap().unwrap();
        assert_eq!(got.id, 7);
        assert!(matches!(
            got.request,
            Request::Transcribe {
                utterance: 3,
                deadline_ms: Some(900),
                ..
            }
        ));
        assert_eq!(bytes_to_pcm(&blob).unwrap(), pcm);
        let (got, blob): (RequestFrame, _) = read_frame(&mut r).unwrap().unwrap();
        assert!(matches!(got.request, Request::Cancel));
        assert!(blob.is_empty());
        assert!(read_frame::<_, RequestFrame>(&mut r).unwrap().is_none());
    }

    #[test]
    fn load_carries_a_pci_id() {
        for gpu in [Some("0000:05:00.0".to_string()), None] {
            let mut buf = Vec::new();
            let req = RequestFrame {
                id: 1,
                request: Request::Load { gpu: gpu.clone() },
            };
            write_frame(&mut buf, &req, &[]).unwrap();
            let (got, _): (RequestFrame, _) = read_frame(&mut buf.as_slice()).unwrap().unwrap();
            let Request::Load { gpu: back } = got.request else {
                panic!("{:?}", got.request);
            };
            assert_eq!(back, gpu);
        }
    }

    #[test]
    fn garbage_and_truncation_are_errors_not_hangs() {
        let mut r: &[u8] = b"hello from a stray printf\n";
        assert!(read_frame::<_, Reply>(&mut r).is_err());
        let mut buf = Vec::new();
        let reply = Reply {
            id: 0,
            body: ReplyBody::Heartbeat,
        };
        write_frame(&mut buf, &reply, &[]).unwrap();
        let mut r = &buf[..buf.len() - 1];
        assert!(read_frame::<_, Reply>(&mut r).is_err());
    }

    #[test]
    fn transcripts_and_errors_survive_the_wire() {
        let t = Transcript {
            utterance: UtteranceId(9),
            text: "Hello there.".into(),
            segments: vec![Segment {
                text: "Hello there.".into(),
                start: Duration::from_millis(80),
                end: Duration::from_millis(900),
            }],
            words: Vec::new(),
            language: Some("en".into()),
            inference_time: Duration::from_micros(41_500),
        };
        let json = serde_json::to_string(&Reply {
            id: 4,
            body: ReplyBody::Transcript((&t).into()),
        })
        .unwrap();
        let back: Reply = serde_json::from_str(&json).unwrap();
        let ReplyBody::Transcript(w) = back.body else {
            panic!("{json}");
        };
        let back: Transcript = w.into();
        assert_eq!(back.utterance, t.utterance);
        assert_eq!(back.segments, t.segments);
        assert_eq!(back.inference_time, t.inference_time);

        for e in [
            SttError::Load("no file".into()),
            SttError::Cancelled,
            SttError::Deadline,
            SttError::EmptyAudio,
        ] {
            let back: SttError = WireError::from(&e).into();
            assert_eq!(back.to_string(), e.to_string());
        }
    }
}
