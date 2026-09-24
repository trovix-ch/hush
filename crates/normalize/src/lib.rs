//! Normalizers, plus the validation that gates every LLM output before it can be
//! inserted.

pub mod chain;
pub mod grammar;
mod lang;
#[cfg(feature = "llama-cpp")]
pub mod llama_cpp;
pub mod llm;
pub mod openai_http;
pub mod prompt;
pub mod rules;
pub mod validate;

pub use chain::{NormalizerChain, should_use_llm, style_allows_llm};
#[cfg(feature = "llama-cpp")]
pub use llama_cpp::{LlamaCppConfig, LlamaCppNormalizer};
pub use openai_http::{HttpConfig, OpenAiHttpNormalizer};
pub use rules::RuleNormalizer;
pub use validate::{Rejection, validate};
