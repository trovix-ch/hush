//! Which GPU speech and the language model run on: the config value, and the policy that
//! turns it into an ordered list of candidate devices. Enumeration lives with the engines;
//! nothing here touches Vulkan.

use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Serialize};

/// `auto`, a PCI bus id (`0000:05:00.0` or `05:00`), or a substring of the device name.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub enum GpuSelector {
    #[default]
    Auto,
    Pci(PciAddress),
    Name(String),
}

impl FromStr for GpuSelector {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, String> {
        let s = s.trim();
        if s.is_empty() {
            return Err(
                "a GPU is \"auto\", a PCI bus id such as \"0000:05:00.0\", or part of \
                        the device name; not empty"
                    .into(),
            );
        }
        if s.eq_ignore_ascii_case("auto") {
            return Ok(Self::Auto);
        }
        if let Some(pci) = PciAddress::parse(s) {
            return Ok(Self::Pci(pci));
        }
        if s.bytes().all(|b| b.is_ascii_digit()) {
            return Err(format!(
                "\"{s}\" is a device index, which names a different card depending on how you \
                 are logged in; use the PCI bus id `hush doctor` prints"
            ));
        }
        Ok(Self::Name(s.to_string()))
    }
}

impl TryFrom<String> for GpuSelector {
    type Error = String;

    fn try_from(s: String) -> Result<Self, String> {
        s.parse()
    }
}

impl From<GpuSelector> for String {
    fn from(s: GpuSelector) -> Self {
        s.to_string()
    }
}

impl fmt::Display for GpuSelector {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Auto => f.write_str("auto"),
            Self::Pci(p) => p.fmt(f),
            Self::Name(n) => f.write_str(n),
        }
    }
}

/// Domain and function are optional in a selector; a device always reports all four.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PciAddress {
    pub domain: Option<u16>,
    pub bus: u8,
    pub device: u8,
    pub function: Option<u8>,
}

impl PciAddress {
    /// `[dddd:]bb:dd[.f]` in hex, fixed widths, so a device name is never mistaken for one.
    pub fn parse(s: &str) -> Option<Self> {
        let hex = |p: &str, width: usize| {
            (p.len() == width && p.bytes().all(|b| b.is_ascii_hexdigit()))
                .then(|| u32::from_str_radix(p, 16).ok())
                .flatten()
        };
        let (rest, function) = match s.rsplit_once('.') {
            Some((r, f)) => (r, Some(u8::try_from(hex(f, 1)?).ok()?)),
            None => (s, None),
        };
        let parts: Vec<&str> = rest.split(':').collect();
        let (domain, bus, device) = match parts.as_slice() {
            [b, d] => (None, *b, *d),
            [dom, b, d] => (Some(u16::try_from(hex(dom, 4)?).ok()?), *b, *d),
            _ => return None,
        };
        Some(Self {
            domain,
            bus: u8::try_from(hex(bus, 2)?).ok()?,
            device: u8::try_from(hex(device, 2)?).ok()?,
            function,
        })
    }

    /// Whether `self`, possibly partial, names the device at `full`.
    pub fn matches(&self, full: &Self) -> bool {
        self.bus == full.bus
            && self.device == full.device
            && self.domain.is_none_or(|d| Some(d) == full.domain)
            && self.function.is_none_or(|f| Some(f) == full.function)
    }
}

impl fmt::Display for PciAddress {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(d) = self.domain {
            write!(f, "{d:04x}:")?;
        }
        write!(f, "{:02x}:{:02x}", self.bus, self.device)?;
        if let Some(func) = self.function {
            write!(f, ".{func:x}")?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum GpuKind {
    Discrete,
    Integrated,
    /// CPU, virtual or unknown.
    Other,
}

/// One Vulkan device as an engine enumerated it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GpuDevice {
    pub description: String,
    /// As the driver reports it, e.g. `0000:05:00.0`.
    pub pci: Option<String>,
    pub kind: GpuKind,
    pub memory_total: u64,
    /// A snapshot: the driver's memory budget minus usage where it reports one (which
    /// counts other processes' allocations), else the total.
    pub memory_free: u64,
}

impl GpuDevice {
    pub fn pci_address(&self) -> Option<PciAddress> {
        self.pci.as_deref().and_then(PciAddress::parse)
    }

    /// The PCI id when there is one, which is what stays stable across sessions.
    pub fn label(&self) -> String {
        match &self.pci {
            Some(p) => format!("{} [{p}]", self.description.trim()),
            None => self.description.trim().to_string(),
        }
    }

    pub fn table_row(&self) -> String {
        let gib = |b: u64| b as f64 / (1u64 << 30) as f64;
        format!(
            "{:<14}{:<12}{:>6.1} GiB total {:>6.1} GiB free  {}",
            self.pci.as_deref().unwrap_or("-"),
            match self.kind {
                GpuKind::Discrete => "discrete",
                GpuKind::Integrated => "integrated",
                GpuKind::Other => "other",
            },
            gib(self.memory_total),
            gib(self.memory_free),
            self.description.trim()
        )
    }
}

/// What a config asks for: a selector, or the index from the deprecated `gpu_device` key.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GpuRequest {
    Selector(GpuSelector),
    LegacyIndex(usize),
}

/// Candidates in the order to try them, as indexes into the device list.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Resolution {
    pub order: Vec<usize>,
    pub why: String,
}

/// Free memory within this much of the best counts as a tie, broken by PCI id, so two
/// idle cards do not swap between sessions over the desktop's few hundred MB; a loaded
/// language model is several GiB and still loses.
pub const FREE_MEMORY_TIE: u64 = 1 << 30;

/// `Err` says what was asked for and what exists.
pub fn resolve(req: &GpuRequest, devices: &[GpuDevice]) -> Result<Resolution, String> {
    let resolution = match req {
        GpuRequest::Selector(GpuSelector::Auto) => auto(devices),
        GpuRequest::Selector(GpuSelector::Pci(want)) => Resolution {
            order: (0..devices.len())
                .filter(|&i| devices[i].pci_address().is_some_and(|a| want.matches(&a)))
                .collect(),
            why: format!("pinned to PCI {want}"),
        },
        GpuRequest::Selector(GpuSelector::Name(name)) => {
            let want = name.to_lowercase();
            let matching: Vec<usize> = (0..devices.len())
                .filter(|&i| devices[i].description.to_lowercase().contains(&want))
                .collect();
            Resolution {
                order: by_free_memory(devices, matching),
                why: format!("name contains {name:?}; most free memory first"),
            }
        }
        GpuRequest::LegacyIndex(i) => Resolution {
            order: if *i < devices.len() {
                vec![*i]
            } else {
                Vec::new()
            },
            why: format!(
                "deprecated gpu_device = {i}; the index order changes between console and \
                 remote sessions"
            ),
        },
    };
    if resolution.order.is_empty() {
        let found: Vec<String> = devices.iter().map(GpuDevice::label).collect();
        return Err(format!(
            "no GPU matches {} (found: {})",
            describe(req),
            if found.is_empty() {
                "none".to_string()
            } else {
                found.join(", ")
            }
        ));
    }
    Ok(resolution)
}

fn describe(req: &GpuRequest) -> String {
    match req {
        GpuRequest::Selector(s) => format!("\"{s}\""),
        GpuRequest::LegacyIndex(i) => format!("index {i}"),
    }
}

/// Discrete cards only when there is one, most free memory first.
fn auto(devices: &[GpuDevice]) -> Resolution {
    let of = |k: GpuKind| (0..devices.len()).filter(move |&i| devices[i].kind == k);
    let discrete: Vec<usize> = of(GpuKind::Discrete).collect();
    if !discrete.is_empty() {
        let skipped = devices.len() - discrete.len();
        return Resolution {
            order: by_free_memory(devices, discrete),
            why: format!(
                "auto: discrete GPU with the most free memory{}",
                match skipped {
                    0 => String::new(),
                    n => format!(", {n} integrated or other skipped"),
                }
            ),
        };
    }
    let mut order = by_free_memory(devices, of(GpuKind::Integrated).collect());
    order.extend(by_free_memory(devices, of(GpuKind::Other).collect()));
    Resolution {
        order,
        why: "auto: no discrete GPU".into(),
    }
}

fn by_free_memory(devices: &[GpuDevice], mut idx: Vec<usize>) -> Vec<usize> {
    let free = |i: usize| devices[i].memory_free;
    let best = idx.iter().map(|&i| free(i)).max().unwrap_or(0);
    let near = |i: usize| free(i).saturating_add(FREE_MEMORY_TIE) >= best;
    idx.sort_by(|&a, &b| {
        near(b)
            .cmp(&near(a))
            .then_with(|| {
                if near(a) && near(b) {
                    std::cmp::Ordering::Equal
                } else {
                    free(b).cmp(&free(a))
                }
            })
            .then_with(|| devices[a].pci.cmp(&devices[b].pci))
            .then(a.cmp(&b))
    });
    idx
}

#[cfg(test)]
mod tests {
    use super::*;

    const GIB: u64 = 1 << 30;

    fn dev(pci: &str, kind: GpuKind, free_gib: u64) -> GpuDevice {
        GpuDevice {
            description: match kind {
                GpuKind::Integrated => "AMD Radeon(TM) Graphics".into(),
                _ => "NVIDIA GeForce RTX 5060 Ti".into(),
            },
            pci: (!pci.is_empty()).then(|| pci.to_string()),
            kind,
            memory_total: 16 * GIB,
            memory_free: free_gib * GIB,
        }
    }

    fn order(req: GpuRequest, devices: &[GpuDevice]) -> Vec<&str> {
        resolve(&req, devices)
            .unwrap()
            .order
            .into_iter()
            .map(|i| devices[i].pci.as_deref().unwrap_or("-"))
            .collect()
    }

    fn sel(s: &str) -> GpuRequest {
        GpuRequest::Selector(s.parse().unwrap())
    }

    #[test]
    fn selectors_parse_in_three_forms_and_never_as_an_index() {
        assert_eq!("auto".parse(), Ok(GpuSelector::Auto));
        assert_eq!(" AUTO ".parse(), Ok(GpuSelector::Auto));
        let full: GpuSelector = "0000:05:00.0".parse().unwrap();
        assert_eq!(
            full,
            GpuSelector::Pci(PciAddress {
                domain: Some(0),
                bus: 5,
                device: 0,
                function: Some(0)
            })
        );
        assert_eq!(full.to_string(), "0000:05:00.0");
        let short: GpuSelector = "05:00".parse().unwrap();
        assert_eq!(short.to_string(), "05:00");
        assert_eq!("0A:1f".parse::<GpuSelector>().unwrap().to_string(), "0a:1f");
        assert_eq!("RTX 5060".parse(), Ok(GpuSelector::Name("RTX 5060".into())));
        assert!(matches!("5:0".parse(), Ok(GpuSelector::Name(_))));
        assert!("1".parse::<GpuSelector>().is_err());
        assert!("".parse::<GpuSelector>().is_err());
    }

    #[test]
    fn a_short_pci_id_matches_the_full_one() {
        let full = PciAddress::parse("0000:05:00.0").unwrap();
        for s in ["05:00", "0000:05:00", "05:00.0", "0000:05:00.0"] {
            assert!(PciAddress::parse(s).unwrap().matches(&full), "{s}");
        }
        for s in ["01:00", "0001:05:00.0", "05:00.1"] {
            assert!(!PciAddress::parse(s).unwrap().matches(&full), "{s}");
        }
    }

    #[test]
    fn auto_drops_the_integrated_gpu_and_prefers_the_most_free_memory() {
        // The console-session order: integrated first, then 05:00, then 01:00 with Ollama.
        let devices = [
            dev("0000:0e:00.0", GpuKind::Integrated, 40),
            dev("0000:05:00.0", GpuKind::Discrete, 15),
            dev("0000:01:00.0", GpuKind::Discrete, 11),
        ];
        assert_eq!(
            order(sel("auto"), &devices),
            ["0000:05:00.0", "0000:01:00.0"]
        );
        let why = resolve(&sel("auto"), &devices).unwrap().why;
        assert!(why.contains("1 integrated"), "{why}");
        // The remote-session order gives the same answer.
        let rdp = [devices[1].clone(), devices[0].clone(), devices[2].clone()];
        assert_eq!(order(sel("auto"), &rdp), ["0000:05:00.0", "0000:01:00.0"]);
    }

    #[test]
    fn auto_breaks_near_ties_by_pci_id_so_idle_cards_do_not_swap() {
        let a = dev("0000:05:00.0", GpuKind::Discrete, 15);
        let mut b = dev("0000:01:00.0", GpuKind::Discrete, 15);
        b.memory_free -= GIB / 4;
        assert_eq!(
            order(sel("auto"), &[a.clone(), b.clone()]),
            ["0000:01:00.0", "0000:05:00.0"]
        );
        b.memory_free = 12 * GIB;
        assert_eq!(
            order(sel("auto"), &[b, a]),
            ["0000:05:00.0", "0000:01:00.0"]
        );
    }

    #[test]
    fn auto_without_a_discrete_gpu_takes_integrated_then_other() {
        let laptop = [
            dev("", GpuKind::Other, 1),
            dev("0000:00:02.0", GpuKind::Integrated, 8),
        ];
        assert_eq!(order(sel("auto"), &laptop), ["0000:00:02.0", "-"]);
        assert!(resolve(&sel("auto"), &[]).is_err());
    }

    #[test]
    fn pinned_selectors_name_one_card_whatever_the_order() {
        let devices = [
            dev("0000:0e:00.0", GpuKind::Integrated, 40),
            dev("0000:05:00.0", GpuKind::Discrete, 2),
            dev("0000:01:00.0", GpuKind::Discrete, 15),
        ];
        assert_eq!(order(sel("05:00"), &devices), ["0000:05:00.0"]);
        assert_eq!(order(sel("0000:05:00.0"), &devices), ["0000:05:00.0"]);
        assert_eq!(order(sel("radeon"), &devices), ["0000:0e:00.0"]);
        assert_eq!(
            order(sel("RTX"), &devices),
            ["0000:01:00.0", "0000:05:00.0"]
        );
        let err = resolve(&sel("09:00"), &devices).unwrap_err();
        assert!(
            err.contains("\"09:00\"") && err.contains("0000:01:00.0"),
            "{err}"
        );
    }

    #[test]
    fn the_deprecated_index_names_the_card_at_that_position() {
        let devices = [
            dev("0000:0e:00.0", GpuKind::Integrated, 40),
            dev("0000:05:00.0", GpuKind::Discrete, 15),
        ];
        assert_eq!(
            order(GpuRequest::LegacyIndex(0), &devices),
            ["0000:0e:00.0"]
        );
        let r = resolve(&GpuRequest::LegacyIndex(1), &devices).unwrap();
        assert!(r.why.contains("deprecated"), "{}", r.why);
        assert!(resolve(&GpuRequest::LegacyIndex(2), &devices).is_err());
    }
}
