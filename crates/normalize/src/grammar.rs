//! GBNF that forbids the output from starting with a preamble word or a quote. It blocks
//! the literal words only; a model determined to preface routes around it, so validation
//! still runs after it.
//!
//! GBNF has no negative lookahead, so the grammar is the complement of a trie of the
//! forbidden prefixes: at each trie node the next char is either one that leaves every
//! forbidden word (then anything may follow) or one that stays on a trie edge. Reaching
//! the end of a forbidden word has no rule, so that path is dead.

use std::collections::BTreeMap;

/// Leading words and characters the grammar forbids unless the source itself starts
/// with them (a dictated "Sure, let's meet" must stay possible).
pub const FORBIDDEN: &[&str] = &[
    "Here",
    "here",
    "Sure",
    "sure",
    "Certainly",
    "certainly",
    "\"",
    "`",
    "\u{201c}",
    "\u{201e}",
    "\u{ab}",
];

pub const ROOT: &str = "root";

pub fn forbidden_for(source: &str) -> Vec<&'static str> {
    let src = source.trim_start().to_lowercase();
    FORBIDDEN
        .iter()
        .copied()
        .filter(|w| !src.starts_with(&w.to_lowercase()))
        .collect()
}

/// The grammar for a request whose rule-pass text is `source`.
pub fn for_source(source: &str) -> String {
    anti_preamble(&forbidden_for(source))
}

pub fn anti_preamble(forbidden: &[&str]) -> String {
    struct Node {
        next: BTreeMap<char, usize>,
        end: bool,
    }
    let mut nodes = vec![Node {
        next: BTreeMap::new(),
        end: false,
    }];
    for w in forbidden {
        let mut cur = 0;
        for c in w.chars() {
            cur = match nodes[cur].next.get(&c) {
                Some(&j) => j,
                None => {
                    nodes.push(Node {
                        next: BTreeMap::new(),
                        end: false,
                    });
                    let j = nodes.len() - 1;
                    nodes[cur].next.insert(c, j);
                    j
                }
            };
        }
        nodes[cur].end = true;
    }

    let mut out = String::new();
    for (i, n) in nodes.iter().enumerate() {
        if n.end {
            continue;
        }
        let mut excluded: Vec<char> = n.next.keys().copied().collect();
        if i == 0 {
            // A leading space or newline would let "\nHere is" through.
            excluded.extend([' ', '\t', '\n', '\r']);
        }
        let class: String = excluded.iter().map(|&c| esc(c)).collect();
        let mut alts = vec![format!("[^{class}] rest")];
        for (&c, &j) in &n.next {
            if !nodes[j].end {
                alts.push(format!("\"{}\" n{j}", esc(c)));
            }
        }
        let body = alts.join(" | ");
        if i == 0 {
            out.push_str(&format!("{ROOT} ::= {body}\n"));
        } else {
            // The output may also end inside a prefix ("He").
            out.push_str(&format!("n{i} ::= ( {body} )?\n"));
        }
    }
    out.push_str("rest ::= [^\\x00]*\n");
    out
}

fn esc(c: char) -> String {
    let u = c as u32;
    if u < 0x80 {
        format!("\\x{u:02X}")
    } else if u <= 0xFFFF {
        format!("\\u{u:04X}")
    } else {
        format!("\\U{u:08X}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_prefix_is_exempt() {
        let f = forbidden_for("  Sure let's meet");
        assert!(!f.contains(&"Sure") && !f.contains(&"sure"));
        assert!(f.contains(&"Here"));
        assert_eq!(forbidden_for("hello").len(), FORBIDDEN.len());
    }

    #[test]
    fn grammar_has_a_rule_per_live_trie_node() {
        let g = anti_preamble(&["Hi", "\""]);
        assert_eq!(
            g,
            "root ::= [^\\x22\\x48\\x20\\x09\\x0A\\x0D] rest | \"\\x48\" n1\n\
             n1 ::= ( [^\\x69] rest )?\n\
             rest ::= [^\\x00]*\n"
        );
    }

    #[test]
    fn shared_prefixes_share_trie_nodes_and_non_ascii_is_escaped() {
        let g = anti_preamble(&["Here", "He", "\u{201c}"]);
        // "He" ends a word, so the node after "e" has no rule and "Here" is unreachable.
        assert!(!g.contains("\\x72"), "{g}");
        assert!(g.contains("\\u201C"), "{g}");
        let full = for_source("hello world");
        assert!(full.starts_with("root ::= [^"));
        assert!(full.ends_with("rest ::= [^\\x00]*\n"));
    }
}
