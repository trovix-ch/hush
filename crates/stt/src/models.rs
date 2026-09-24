//! Model manifest and download.
//!
//! The manifest is compiled into the binary so the set of URLs the app may fetch is fixed
//! at build time; nothing outside it is ever downloaded.

use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::time::{Duration, Instant};

use serde::Deserialize;
use sha2::{Digest, Sha256};

const MANIFEST_JSON: &str = include_str!("models.json");

/// Suffix of a file that is still being downloaded. A file without it has passed the size
/// and hash checks, so presence alone is trusted on later starts: re-hashing 2.4 GB on
/// every launch would cost seconds of startup.
const PART_SUFFIX: &str = ".part";

#[derive(Debug, thiserror::Error)]
pub enum ModelError {
    #[error("model manifest is invalid: {0}")]
    Manifest(String),
    #[error("unknown model id `{0}`")]
    UnknownModel(String),
    #[error("could not determine the local application data directory")]
    NoDataDir,
    #[error("I/O error on {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: io::Error,
    },
    #[error("download of {url} failed: {message}")]
    Http { url: String, message: String },
    #[error("{name}: expected {expected} bytes, got {actual}")]
    SizeMismatch {
        name: String,
        expected: u64,
        actual: u64,
    },
    #[error("{name}: SHA-256 mismatch, expected {expected}, got {actual}")]
    HashMismatch {
        name: String,
        expected: String,
        actual: String,
    },
}

fn io_err(path: &Path) -> impl FnOnce(io::Error) -> ModelError + '_ {
    move |source| ModelError::Io {
        path: path.to_path_buf(),
        source,
    }
}

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ModelFile {
    /// File name inside the model directory.
    pub name: String,
    pub url: String,
    /// Exact size in bytes.
    pub size: u64,
    /// Lower-case hex SHA-256, when known.
    pub sha256: Option<String>,
}

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ModelManifest {
    /// Stable id, also the directory name under the models root.
    pub id: String,
    /// Which engine implementation can load these files.
    pub engine: String,
    pub description: String,
    /// SPDX identifier.
    pub license: String,
    /// Text that must be shown to the user where the license requires attribution.
    pub attribution: String,
    pub files: Vec<ModelFile>,
}

impl ModelManifest {
    pub fn total_size(&self) -> u64 {
        self.files.iter().map(|f| f.size).sum()
    }

    /// True when every file is present at its final name with the expected size.
    pub fn is_present(&self, dir: &Path) -> bool {
        self.files.iter().all(|f| {
            fs::metadata(dir.join(&f.name))
                .map(|m| m.len() == f.size)
                .unwrap_or(false)
        })
    }

    /// Path of the file an engine opens: the single file of a one-file model, or the
    /// directory for multi-file models (ONNX encoder + joint + vocab).
    pub fn load_path(&self, dir: &Path) -> PathBuf {
        match self.files.as_slice() {
            [only] => dir.join(&only.name),
            _ => dir.to_path_buf(),
        }
    }
}

/// Id of the model a default build uses.
pub const DEFAULT_MODEL_ID: &str = "parakeet-tdt-0.6b-v3-f16-gguf";

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ManifestDoc {
    models: Vec<ModelManifest>,
}

/// Parse and validate a manifest document.
pub fn parse_manifest(json: &str) -> Result<Vec<ModelManifest>, ModelError> {
    let doc: ManifestDoc =
        serde_json::from_str(json).map_err(|e| ModelError::Manifest(e.to_string()))?;
    let mut seen = std::collections::HashSet::new();
    for m in &doc.models {
        if m.id.is_empty() || !is_safe_file_name(&m.id) {
            return Err(ModelError::Manifest(format!("bad model id `{}`", m.id)));
        }
        if !seen.insert(m.id.as_str()) {
            return Err(ModelError::Manifest(format!(
                "duplicate model id `{}`",
                m.id
            )));
        }
        if m.files.is_empty() {
            return Err(ModelError::Manifest(format!(
                "model `{}` has no files",
                m.id
            )));
        }
        for f in &m.files {
            if !is_safe_file_name(&f.name) {
                return Err(ModelError::Manifest(format!(
                    "model `{}`: unsafe file name `{}`",
                    m.id, f.name
                )));
            }
            if !f.url.starts_with("https://") {
                return Err(ModelError::Manifest(format!(
                    "model `{}`: `{}` is not an https URL",
                    m.id, f.url
                )));
            }
            if let Some(h) = &f.sha256
                && (h.len() != 64 || !h.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f')))
            {
                return Err(ModelError::Manifest(format!(
                    "model `{}`: `{}` has a malformed sha256",
                    m.id, f.name
                )));
            }
        }
    }
    Ok(doc.models)
}

/// A manifest file name must not escape the model directory.
fn is_safe_file_name(name: &str) -> bool {
    !name.is_empty()
        && name != "."
        && name != ".."
        && !name.contains(['/', '\\', ':'])
        && !name.ends_with(PART_SUFFIX)
}

/// Every model this build knows about.
pub fn manifest() -> &'static [ModelManifest] {
    static MODELS: OnceLock<Vec<ModelManifest>> = OnceLock::new();
    MODELS.get_or_init(|| parse_manifest(MANIFEST_JSON).expect("embedded manifest is valid"))
}

pub fn find(id: &str) -> Result<&'static ModelManifest, ModelError> {
    manifest()
        .iter()
        .find(|m| m.id == id)
        .ok_or_else(|| ModelError::UnknownModel(id.to_string()))
}

/// `%LOCALAPPDATA%\whisper-local\models` on Windows. Local, not Roaming: these are
/// gigabytes and must never sync with a roaming profile.
pub fn default_models_root() -> Result<PathBuf, ModelError> {
    let dirs =
        directories::ProjectDirs::from("", "", "whisper-local").ok_or(ModelError::NoDataDir)?;
    Ok(models_root_from_data_local(dirs.data_local_dir()))
}

/// `ProjectDirs` appends a `data` component on Windows only; strip it so the models sit
/// beside the app's other folders rather than inside `data`.
fn models_root_from_data_local(data_local: &Path) -> PathBuf {
    let base = if cfg!(windows) && data_local.file_name().is_some_and(|n| n == "data") {
        data_local.parent().unwrap_or(data_local)
    } else {
        data_local
    };
    base.join("models")
}

pub fn default_model_dir(model_id: &str) -> Result<PathBuf, ModelError> {
    Ok(default_models_root()?.join(model_id))
}

/// Download every missing file of `model` into `dir`, resuming partial downloads, and
/// verify size and hash. Blocking. Returns immediately when everything is present.
pub fn ensure_downloaded(model: &ModelManifest, dir: &Path) -> Result<(), ModelError> {
    fs::create_dir_all(dir).map_err(io_err(dir))?;
    let agent: ureq::Agent = ureq::Agent::config_builder()
        .timeout_connect(Some(Duration::from_secs(30)))
        .timeout_recv_response(Some(Duration::from_secs(60)))
        .build()
        .into();
    for file in &model.files {
        let dest = dir.join(&file.name);
        if fs::metadata(&dest).is_ok_and(|m| m.len() == file.size) {
            continue;
        }
        tracing::info!(model = %model.id, file = %file.name, size = file.size, "downloading");
        download_file(&agent, file, &dest)?;
    }
    Ok(())
}

fn part_path(dest: &Path) -> PathBuf {
    let mut s = dest.as_os_str().to_owned();
    s.push(PART_SUFFIX);
    PathBuf::from(s)
}

fn download_file(agent: &ureq::Agent, file: &ModelFile, dest: &Path) -> Result<(), ModelError> {
    let part = part_path(dest);
    let mut have = fs::metadata(&part).map(|m| m.len()).unwrap_or(0);
    if have > file.size {
        fs::remove_file(&part).map_err(io_err(&part))?;
        have = 0;
    }

    if have < file.size {
        let http_err = |message: String| ModelError::Http {
            url: file.url.clone(),
            message,
        };
        let mut req = agent.get(&file.url);
        if have > 0 {
            req = req.header("Range", format!("bytes={have}-"));
        }
        let resp = req.call().map_err(|e| http_err(e.to_string()))?;
        let status = resp.status().as_u16();
        let append = match status {
            206 => true,
            // A server that ignores Range sends the whole body; start over rather than
            // appending a second copy.
            200 => false,
            s => return Err(http_err(format!("unexpected HTTP status {s}"))),
        };
        if have > 0 {
            tracing::info!(file = %file.name, resumed_at = have, append, "resuming");
        }
        let mut out = OpenOptions::new()
            .create(true)
            .write(true)
            .append(append)
            .truncate(!append)
            .open(&part)
            .map_err(io_err(&part))?;
        if !append {
            have = 0;
        }
        let mut reader = resp.into_body().into_reader();
        copy_with_progress(&mut reader, &mut out, file, have, &part)?;
        out.sync_all().map_err(io_err(&part))?;
    }

    let actual = fs::metadata(&part).map_err(io_err(&part))?.len();
    if actual != file.size {
        return Err(ModelError::SizeMismatch {
            name: file.name.clone(),
            expected: file.size,
            actual,
        });
    }
    if let Some(expected) = &file.sha256 {
        let actual = sha256_file(&part)?;
        if &actual != expected {
            // A corrupt partial file would otherwise be resumed forever.
            let _ = fs::remove_file(&part);
            return Err(ModelError::HashMismatch {
                name: file.name.clone(),
                expected: expected.clone(),
                actual,
            });
        }
    }
    fs::rename(&part, dest).map_err(io_err(dest))?;
    Ok(())
}

fn copy_with_progress(
    reader: &mut impl Read,
    out: &mut File,
    file: &ModelFile,
    start: u64,
    part: &Path,
) -> Result<(), ModelError> {
    let mut buf = vec![0u8; 1 << 20];
    let mut done = start;
    let began = Instant::now();
    let mut last_log = Instant::now();
    loop {
        let n = reader.read(&mut buf).map_err(|e| ModelError::Http {
            url: file.url.clone(),
            message: e.to_string(),
        })?;
        if n == 0 {
            break;
        }
        out.write_all(&buf[..n]).map_err(io_err(part))?;
        done += n as u64;
        if last_log.elapsed() >= Duration::from_secs(5) {
            last_log = Instant::now();
            let secs = began.elapsed().as_secs_f64().max(1e-3);
            tracing::info!(
                file = %file.name,
                percent = format!("{:.1}", done as f64 * 100.0 / file.size.max(1) as f64),
                mib_per_s = format!("{:.1}", (done - start) as f64 / secs / 1_048_576.0),
                "download progress"
            );
        }
    }
    Ok(())
}

/// Lower-case hex SHA-256 of a file.
pub fn sha256_file(path: &Path) -> Result<String, ModelError> {
    let mut f = File::open(path).map_err(io_err(path))?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = f.read(&mut buf).map_err(io_err(path))?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(hex(&hasher.finalize()))
}

fn hex(bytes: &[u8]) -> String {
    use std::fmt::Write as _;
    bytes
        .iter()
        .fold(String::with_capacity(bytes.len() * 2), |mut s, b| {
            let _ = write!(s, "{b:02x}");
            s
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedded_manifest_parses_and_has_parakeet() {
        let m = find("parakeet-tdt-0.6b-v3").unwrap();
        assert_eq!(m.license, "CC-BY-4.0");
        assert!(m.attribution.contains("NVIDIA"));
        assert!(m.files.iter().any(|f| f.name == "vocab.txt"));
        assert!(m.files.iter().all(|f| f.sha256.is_some()));
        assert!(find("parakeet-tdt-0.6b-v3-int8").is_ok());
        assert!(matches!(find("nope"), Err(ModelError::UnknownModel(_))));
    }

    #[test]
    fn default_model_is_a_single_gguf_with_hash() {
        let m = find(DEFAULT_MODEL_ID).unwrap();
        assert_eq!(m.engine, "transcribe-cpp");
        assert_eq!(m.files.len(), 1);
        assert!(m.files[0].sha256.is_some());
        let dir = Path::new("root");
        assert_eq!(m.load_path(dir), dir.join("parakeet-tdt-0.6b-v3-F16.gguf"));
        assert!(find("whisper-large-v3-turbo-f16-gguf").is_ok());
        assert_eq!(find("parakeet-tdt-0.6b-v3").unwrap().load_path(dir), dir);
    }

    #[test]
    fn every_url_is_pinned_to_a_revision() {
        for m in manifest() {
            for f in &m.files {
                assert!(
                    !f.url.contains("/resolve/main/"),
                    "{} floats on main",
                    f.url
                );
                assert!(f.url.ends_with(&format!("/{}", f.name)), "{}", f.url);
            }
        }
    }

    fn one_model(file: &str) -> String {
        format!(
            r#"{{"models":[{{"id":"m","engine":"e","description":"d","license":"MIT",
            "attribution":"a","files":[{file}]}}]}}"#
        )
    }

    #[test]
    fn rejects_bad_entries() {
        let ok = r#"{"name":"a.onnx","url":"https://x/a.onnx","size":1,"sha256":null}"#;
        assert!(parse_manifest(&one_model(ok)).is_ok());

        let cases = [
            r#"{"name":"../a","url":"https://x/a","size":1,"sha256":null}"#,
            r#"{"name":"a\\b","url":"https://x/a","size":1,"sha256":null}"#,
            r#"{"name":"a.part","url":"https://x/a","size":1,"sha256":null}"#,
            r#"{"name":"a","url":"http://x/a","size":1,"sha256":null}"#,
            r#"{"name":"a","url":"https://x/a","size":1,"sha256":"ABC"}"#,
            r#"{"name":"a","url":"https://x/a","size":1,"sha256":null,"extra":1}"#,
        ];
        for c in cases {
            assert!(parse_manifest(&one_model(c)).is_err(), "accepted {c}");
        }
    }

    #[test]
    fn rejects_duplicate_ids() {
        let m = r#"{"id":"m","engine":"e","description":"d","license":"MIT","attribution":"a",
            "files":[{"name":"a","url":"https://x/a","size":1,"sha256":null}]}"#;
        let json = format!(r#"{{"models":[{m},{m}]}}"#);
        assert!(matches!(
            parse_manifest(&json),
            Err(ModelError::Manifest(_))
        ));
    }

    #[test]
    fn models_root_strips_windows_data_component() {
        let root = models_root_from_data_local(Path::new("base/whisper-local/data"));
        if cfg!(windows) {
            assert_eq!(root, Path::new("base/whisper-local/models"));
        } else {
            assert_eq!(root, Path::new("base/whisper-local/data/models"));
        }
        let root = models_root_from_data_local(Path::new("base/whisper-local"));
        assert_eq!(root, Path::new("base/whisper-local/models"));
    }

    #[cfg(windows)]
    #[test]
    fn default_model_dir_is_under_local_app_data() {
        let dir = default_model_dir("parakeet-tdt-0.6b-v3").unwrap();
        assert!(dir.ends_with(r"whisper-local\models\parakeet-tdt-0.6b-v3"));
        if let Some(local) = std::env::var_os("LOCALAPPDATA") {
            assert!(dir.starts_with(local));
        }
    }

    #[test]
    fn presence_requires_exact_sizes() {
        let tmp = tempfile::tempdir().unwrap();
        let json = one_model(r#"{"name":"a.bin","url":"https://x/a.bin","size":3,"sha256":null}"#);
        let m = &parse_manifest(&json).unwrap()[0];
        assert!(!m.is_present(tmp.path()));
        fs::write(tmp.path().join("a.bin"), b"ab").unwrap();
        assert!(!m.is_present(tmp.path()));
        fs::write(tmp.path().join("a.bin"), b"abc").unwrap();
        assert!(m.is_present(tmp.path()));
    }

    #[test]
    fn sha256_matches_known_vector() {
        let tmp = tempfile::tempdir().unwrap();
        let p = tmp.path().join("x");
        fs::write(&p, b"abc").unwrap();
        assert_eq!(
            sha256_file(&p).unwrap(),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }
}
