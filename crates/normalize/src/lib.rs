//! Normalizers implementing `wl_core::normalize::Normalizer`, plus the validation that
//! gates every LLM output before it can be inserted.

pub mod chain;
mod lang;
pub mod openai_http;
pub mod prompt;
pub mod rules;
pub mod validate;

pub use chain::{NormalizerChain, should_use_llm};
pub use openai_http::{HttpConfig, OpenAiHttpNormalizer};
pub use rules::RuleNormalizer;
pub use validate::{Rejection, validate};
