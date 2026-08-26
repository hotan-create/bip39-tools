use anyhow::{Context, Result};
use bip39::{Language, Mnemonic};
use bitcoin::address::{Address, NetworkChecked, NetworkUnchecked};
use bitcoin::bip32::{DerivationPath, Xpriv};
use bitcoin::secp256k1::{Secp256k1, SecretKey};
use bitcoin::{Network, PublicKey};
use clap::Parser;
use itertools::Itertools;
use rayon::iter::ParallelBridge;
use rayon::prelude::*;
use sha3::{Digest, Keccak256};
use std::collections::HashSet;
use std::fs;
use std::io::{self, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

mod candidates;
mod gpu;

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

#[derive(Parser, Debug)]
#[command(
    about = "BIP-39 mnemonic recovery — word mode or tokenlist mode.\n\
             Word mode   : supply 10-12 known words as positional args.\n\
             Tokenlist   : supply --tokenlist <file> (see --help for format).",
    version
)]
struct Args {
    /// Target Bitcoin legacy address (Base58, starts with '1').
    target_address: Option<String>,

    /// Known words (word mode only). 10, 11, or 12 words.
    /// A word wrapped in literal double quotes (e.g. "tornado") is pinned:
    /// kept at its position, excluded from permutation.
    /// Omit when using --tokenlist.
    words: Vec<String>,

    /// Path to tokenlist file (tokenlist mode).
    ///
    /// FORMAT
    /// ======
    /// • One line = one SLOT (positional group of words).
    /// • Blank lines and '#' comments are ignored.
    /// • Alternatives within a line separated by whitespace.
    /// • Words within an alternative separated by commas.
    /// • '?' = unknown word, brute-forced from full BIP-39 list. Its
    ///   *position* is permuted along with the known words too (unless
    ///   --keep-word-order), so you don't need one alternative per '?' slot.
    /// • "word" (in literal double quotes) = pinned: kept at its position,
    ///   excluded from permutation, even without --keep-word-order.
    /// • "?" (literal double quotes around '?') = unknown word whose
    ///   *position* is pinned, but the value is still brute-forced.
    ///
    /// EXAMPLE
    ///   zebra,"tornado",gravity,?   abandon,art   <- slot 1 (2 alternatives)
    ///   orbit,galaxy                               <- slot 2
    ///   venture,sun                                <- slot 3
    /// (in alt 1: tornado stays 2nd; zebra/gravity/? all permute among the
    ///  other 3 positions — no need to write '?' in every possible slot)
    ///
    /// Total words across chosen slots must equal 12.
    #[arg(long, value_name = "FILE")]
    tokenlist: Option<PathBuf>,

    /// Keep slot order as written (no slot permutations).
    #[arg(long)]
    keep_token_order: bool,

    /// Keep word order within each slot (tokenlist mode; no intra-slot
    /// permutations). Not used in word mode — there, pin individual words
    /// with literal double quotes instead (see the positional args above).
    #[arg(long)]
    keep_word_order: bool,

    /// Minimum number of slots to use (default: all).
    #[arg(long, value_name = "N")]
    min_token: Option<usize>,

    /// BIP-39 wordlist language.
    #[arg(long, short, default_value = "english")]
    language: String,

    /// Target coin: btc (P2PKH, default) or eth.
    #[arg(long, default_value = "btc")]
    coin: String,

    /// Override BIP-32 derivation path.
    /// Defaults: m/44'/0'/0'/0/0 (btc), m/44'/60'/0'/0/0 (eth).
    #[arg(long, value_name = "PATH")]
    derivation_path: Option<String>,

    /// Number of CPU threads (0 = all cores).
    #[arg(long, short, default_value_t = 0)]
    threads: usize,

    /// Verify GPU crypto primitives against CPU reference then exit.
    #[arg(long)]
    selftest: bool,

    /// Force CPU search (skip GPU even if available).
    #[arg(long)]
    cpu: bool,

    /// Override GPU batch size exactly.
    #[arg(long, value_name = "N")]
    batch_size: Option<usize>,

    /// Probe START: batch = 2^EXP (default 16 = 65 536).
    #[arg(long, value_name = "EXP", default_value_t = 16)]
    min_batch: u32,

    /// Probe CAP: batch never exceeds 2^EXP (default 28 = 268M).
    /// Use --max-batch 16 for 2 GB VRAM, --max-batch 17 for 4 GB, etc.
    #[arg(long, value_name = "EXP", default_value_t = 28)]
    max_batch: u32,
}

// ---------------------------------------------------------------------------
// Tokenlist data model
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
enum Token {
    Word(String),
    /// A word wrapped in literal double quotes, e.g. `"tornado"` — kept at
    /// its position, excluded from permutation (unless --keep-word-order
    /// already made permutation a no-op).
    PinnedWord(String),
    /// `?` — unknown word, brute-forced from the full wordlist. By default
    /// its *position* is also permuted along with the known words (so you
    /// don't have to write out one alternative per possible '?' slot).
    Missing,
    /// `"?"` — unknown word whose *position* is pinned (kept exactly where
    /// written), but the value is still brute-forced.
    PinnedMissing,
}

type Alternative = Vec<Token>;
type Slot        = Vec<Alternative>;

/// A word wrapped in literal double quotes (`"..."`, length >= 2) marks it
/// pinned. Shared by tokenlist parsing and word-mode CLI args.
fn is_pinned_token(s: &str) -> bool {
    s.len() >= 2 && s.starts_with('"') && s.ends_with('"')
}

fn strip_pin(s: &str) -> &str {
    if is_pinned_token(s) { &s[1..s.len() - 1] } else { s }
}

// ---------------------------------------------------------------------------
// Coin / Target
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Coin {
    Btc,
    Eth,
}

fn parse_coin(s: &str) -> Result<Coin> {
    match s.to_lowercase().as_str() {
        "btc" | "bitcoin"  => Ok(Coin::Btc),
        "eth" | "ethereum" => Ok(Coin::Eth),
        _ => anyhow::bail!("Unknown coin '{s}' (expected 'btc' or 'eth')"),
    }
}

fn default_derivation_path(coin: Coin) -> &'static str {
    match coin {
        Coin::Btc => "m/44'/0'/0'/0/0",
        Coin::Eth => "m/44'/60'/0'/0/0",
    }
}

/// Parsed recovery target: a Bitcoin address or a normalized (lowercase,
/// 0x-prefixed) Ethereum address string.
#[derive(Debug, Clone)]
enum Target {
    Btc(Address<NetworkChecked>),
    Eth(String),
}

impl Target {
    fn as_display_string(&self) -> String {
        match self {
            Target::Btc(a) => a.to_string(),
            Target::Eth(s) => eth_checksum_address(s),
        }
    }

    /// Lowercase comparison string (checksum-insensitive).
    fn as_compare_string(&self) -> String {
        match self {
            Target::Btc(a) => a.to_string(),
            Target::Eth(s) => s.clone(),
        }
    }

}

/// The 20-byte value candidates are compared against: hash160 for BTC,
/// the raw ETH address bytes for ETH.
fn target_hash20(target: &Target) -> Result<[u8; 20]> {
    match target {
        Target::Btc(a) => p2pkh_hash160(a),
        Target::Eth(s) => {
            let body = &s[2..]; // strip "0x"
            let mut out = [0u8; 20];
            for i in 0..20 {
                out[i] = u8::from_str_radix(&body[i*2..i*2+2], 16)
                    .with_context(|| format!("invalid ETH address hex: {s}"))?;
            }
            Ok(out)
        }
    }
}

/// GPU pipeline kernel name for a coin (see kernels.cu).
fn pipeline_kernel(coin: Coin) -> &'static str {
    match coin {
        Coin::Btc => "k_pipeline",
        Coin::Eth => "k_pipeline_eth",
    }
}

/// Normalize a user-supplied ETH address to lowercase "0x" + 40 hex chars.
/// Accepts either checksummed or all-lowercase/uppercase input.
fn normalize_eth_address(s: &str) -> Result<String> {
    let body = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")).unwrap_or(s);
    anyhow::ensure!(body.len() == 40, "Invalid ETH address (expected 40 hex chars): {s}");
    anyhow::ensure!(body.chars().all(|c| c.is_ascii_hexdigit()), "Invalid ETH address (non-hex chars): {s}");
    Ok(format!("0x{}", body.to_lowercase()))
}

/// Apply EIP-55 mixed-case checksum to a lowercase "0x..." address, for display only.
fn eth_checksum_address(lower: &str) -> String {
    let body = &lower[2..];
    let hash = Keccak256::digest(body.as_bytes());
    let mut out = String::with_capacity(42);
    out.push_str("0x");
    for (i, c) in body.chars().enumerate() {
        if c.is_ascii_digit() {
            out.push(c);
            continue;
        }
        let byte = hash[i / 2];
        let nibble = if i % 2 == 0 { byte >> 4 } else { byte & 0x0f };
        if nibble >= 8 { out.push(c.to_ascii_uppercase()); } else { out.push(c); }
    }
    out
}

/// Derive the lowercase "0x..." ETH address from a child private key.
fn eth_address_from_privkey(secret: &SecretKey, secp: &Secp256k1<bitcoin::secp256k1::All>) -> String {
    let pubkey = bitcoin::secp256k1::PublicKey::from_secret_key(secp, secret);
    let uncompressed = pubkey.serialize_uncompressed(); // [0x04, X(32), Y(32)] = 65 bytes
    let hash = Keccak256::digest(&uncompressed[1..]);    // hash the 64-byte X||Y
    let addr_bytes = &hash[12..]; // last 20 bytes
    format!("0x{}", hex_lower(addr_bytes))
}

fn hex_lower(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes { s.push_str(&format!("{:02x}", b)); }
    s
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

fn main() -> Result<()> {
    let args = Args::parse();

    if args.selftest {
        println!("Running GPU primitive selftests...");
        let ok = gpu::run_selftest()?;
        if ok { println!("All selftests passed."); std::process::exit(0); }
        else  { eprintln!("One or more selftests FAILED"); std::process::exit(1); }
    }

    let coin = parse_coin(&args.coin)?;
    let target_str = args.target_address.as_deref().context("Missing target address")?;
    let target: Target = match coin {
        Coin::Btc => {
            let addr: Address<NetworkChecked> = target_str
                .parse::<Address<NetworkUnchecked>>().context("Invalid Bitcoin address")?
                .require_network(Network::Bitcoin.into())
                .context("Only mainnet legacy addresses supported")?;
            Target::Btc(addr)
        }
        Coin::Eth => Target::Eth(normalize_eth_address(target_str)?),
    };

    let deriv_path_str = args.derivation_path.clone().unwrap_or_else(|| default_derivation_path(coin).to_string());
    let deriv: DerivationPath = deriv_path_str.parse().context("Invalid derivation path")?;
    // The CUDA kernels only implement each coin's fixed default path; a
    // custom --derivation-path can only run on CPU.
    let custom_path = args.derivation_path.is_some() && deriv_path_str != default_derivation_path(coin);

    let language = parse_language(&args.language)?;
    let wall     = Instant::now();

    let force_cpu = args.cpu || custom_path;
    if custom_path && !args.cpu {
        println!("Custom --derivation-path: GPU kernels only support each coin's default path, using CPU.");
    }

    let found = if args.tokenlist.is_some() {
        // ── tokenlist mode ──────────────────────────────────────────────────
        let path  = args.tokenlist.as_ref().unwrap();
        let slots = parse_tokenlist(path)?;
        validate_slots(&slots, language)?;

        if force_cpu {
            if !custom_path { println!("--cpu: using CPU (tokenlist mode)."); }
            run_tokenlist_cpu(&args, &slots, &target, language, coin, &deriv)?
        } else {
            match run_tokenlist_gpu(&args, &slots, &target, language, coin) {
                Ok(f) => f,
                Err(e) => {
                    eprintln!("GPU unavailable ({e:#}); falling back to CPU.");
                    run_tokenlist_cpu(&args, &slots, &target, language, coin, &deriv)?
                }
            }
        }
    } else {
        // ── word mode ───────────────────────────────────────────────────────
        if !(10..=12).contains(&args.words.len()) {
            anyhow::bail!("Word mode: expected 10–12 words, got {}. Use --tokenlist for slot-based search.", args.words.len());
        }
        if force_cpu {
            run_word_cpu(&args, &target, language, coin, &deriv)?
        } else {
            match run_word_gpu(&args, &target, language, coin) {
                Ok(f) => f,
                Err(e) => {
                    eprintln!("GPU unavailable ({e:#}); falling back to CPU.");
                    run_word_cpu(&args, &target, language, coin, &deriv)?
                }
            }
        }
    };

    if !found {
        println!(
            "\nNo match found.  Elapsed: {:.3}s",
            wall.elapsed().as_secs_f64()
        );
    }
    Ok(())
}

// ===========================================================================
// TOKENLIST PARSING
// ===========================================================================

fn parse_tokenlist(path: &PathBuf) -> Result<Vec<Slot>> {
    let content = fs::read_to_string(path)
        .with_context(|| format!("Cannot read tokenlist: {}", path.display()))?;

    let mut slots: Vec<Slot> = Vec::new();
    for (lineno, raw) in content.lines().enumerate() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') { continue; }

        let alternatives: Vec<Alternative> = line
            .split_whitespace()
            .map(|alt_str| {
                alt_str.split(',')
                    .filter(|t| !t.is_empty())
                    .map(|t| {
                        let t = t.trim();
                        if t == "?" {
                            Token::Missing
                        } else if is_pinned_token(t) {
                            if strip_pin(t) == "?" {
                                Token::PinnedMissing
                            } else {
                                Token::PinnedWord(strip_pin(t).to_string())
                            }
                        } else {
                            Token::Word(t.to_string())
                        }
                    })
                    .collect::<Alternative>()
            })
            .filter(|a| !a.is_empty())
            .collect();

        if alternatives.is_empty() {
            eprintln!("Warning: line {} empty after parsing, skipping.", lineno + 1);
            continue;
        }
        slots.push(alternatives);
    }

    anyhow::ensure!(!slots.is_empty(), "Tokenlist is empty or has no valid lines");
    println!("Loaded {} slot(s) from tokenlist.", slots.len());
    Ok(slots)
}

fn validate_slots(slots: &[Slot], language: Language) -> Result<()> {
    let wl: &'static [&'static str] = language.words_by_prefix("");
    for (si, slot) in slots.iter().enumerate() {
        for (ai, alt) in slot.iter().enumerate() {
            for tok in alt {
                if let Token::Word(w) | Token::PinnedWord(w) = tok {
                    anyhow::ensure!(
                        wl.contains(&w.as_str()),
                        "Slot {}, alt {}: '{}' not in BIP-39 wordlist", si+1, ai+1, w
                    );
                }
            }
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// AltMeta / nth_permutation / nth_combination / nth_missing_combo: lazily
// materialize one alternative's partial phrase for a given flat candidate
// index — O(len) or O(len^2), never materializing the other permutations,
// combinations, or wl_len^k combos upfront.
// ---------------------------------------------------------------------------

/// The `idx`-th (0-based) permutation of `items`, via the factorial number
/// system (Lehmer code). O(n^2) worst case (n small in practice: <=12).
fn nth_permutation(items: &[u16], mut idx: usize) -> Vec<u16> {
    let mut pool: Vec<u16> = items.to_vec();
    let n = pool.len();
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let f   = factorial(n - 1 - i);
        let sel = idx / f;
        idx %= f;
        out.push(pool.remove(sel));
    }
    out
}

fn binomial(n: usize, k: usize) -> usize {
    if k > n { return 0; }
    let k = k.min(n - k);
    let mut result: usize = 1;
    for i in 0..k {
        result = result * (n - i) / (i + 1);
    }
    result
}

/// The `idx`-th (0-based, lexicographic) k-combination of {0, ..., n-1},
/// via the combinatorial number system. O(n*k) worst case (n, k small).
fn nth_combination(n: usize, k: usize, mut idx: usize) -> Vec<usize> {
    let mut result = Vec::with_capacity(k);
    let mut start = 0usize;
    for i in 0..k {
        let remaining = k - i - 1;
        let mut c = start;
        loop {
            let cnt = binomial(n - c - 1, remaining);
            if idx < cnt {
                result.push(c);
                start = c + 1;
                break;
            }
            idx -= cnt;
            c += 1;
        }
    }
    result
}

/// The `idx`-th (0-based) of the `wl_len^k` combinations-with-replacement,
/// as a k-digit base-`wl_len` number.
fn nth_missing_combo(wl_len: u16, k: usize, mut idx: usize) -> Vec<u16> {
    let mut out = vec![0u16; k];
    for i in (0..k).rev() {
        out[i] = (idx % wl_len as usize) as u16;
        idx /= wl_len as usize;
    }
    out
}

/// Per-alternative metadata: enough to compute the `idx`-th candidate word
/// sequence on demand, without ever generating the other candidates for
/// this alternative. This is what makes `LazyPhraseIter` actually lazy —
/// previously this was a fully-materialized `Vec<Vec<u16>>` per
/// alternative, which meant e.g. 20,000 alternatives x 7! permutations
/// each had to be built and held in RAM before the search could start.
///
/// Four kinds of token, per position:
///   Word          — movable: known value, position permutes freely
///   Missing (?)   — movable: unknown value (brute-forced), position also
///                   permutes freely (mixed in with the known words)
///   PinnedWord    — fixed position AND value
///   PinnedMissing (") — fixed position, unknown value (still brute-forced)
///
/// The `m` known movable words and `q` movable '?' marks are permuted
/// together as one multiset of `m+q` slots (the '?' marks are
/// interchangeable among themselves for *position* purposes — each still
/// gets its own independently brute-forced value).
struct AltMeta {
    total_len:         usize,
    pinned_fixed:      Vec<(usize, u16)>,
    pinned_missing:    Vec<usize>,
    movable_known:     Vec<u16>,     // in original relative order
    movable_positions: Vec<usize>,   // the m+q position indices (ascending)
    movable_is_missing: Vec<bool>,   // original layout of those m+q slots (used by --keep-word-order)
    movable_missing:   usize,        // q
    known_perm_count:  usize,        // m!
    perm_count:        usize,        // C(m+q, q) * m!  (or 1 with --keep-word-order)
    combo_count:       usize,        // wl_len ^ (q + pinned_missing.len())
}

impl AltMeta {
    fn new(alt: &Alternative, wordlist: &'static [&'static str], wl_len: usize, keep_word_order: bool) -> Self {
        let mut pinned_fixed:       Vec<(usize, u16)> = Vec::new();
        let mut pinned_missing:     Vec<usize> = Vec::new();
        let mut movable_known:      Vec<u16> = Vec::new();
        let mut movable_positions:  Vec<usize> = Vec::new();
        let mut movable_is_missing: Vec<bool> = Vec::new();

        for (i, tok) in alt.iter().enumerate() {
            match tok {
                Token::Word(w) => {
                    movable_known.push(wordlist.iter().position(|x| *x == w.as_str()).unwrap() as u16);
                    movable_positions.push(i);
                    movable_is_missing.push(false);
                }
                Token::Missing => {
                    movable_positions.push(i);
                    movable_is_missing.push(true);
                }
                Token::PinnedWord(w) => pinned_fixed.push((i, wordlist.iter().position(|x| *x == w.as_str()).unwrap() as u16)),
                Token::PinnedMissing => pinned_missing.push(i),
            }
        }

        let m = movable_known.len();
        let q = movable_is_missing.iter().filter(|&&is_q| is_q).count();
        let known_perm_count = if keep_word_order { 1 } else { factorial(m) };
        let perm_count = if keep_word_order { 1 } else { binomial(m + q, q) * known_perm_count };
        let combo_count = wl_len.pow((q + pinned_missing.len()) as u32).max(1);

        AltMeta {
            total_len: alt.len(),
            pinned_fixed, pinned_missing,
            movable_known, movable_positions, movable_is_missing,
            movable_missing: q,
            known_perm_count, perm_count, combo_count,
        }
    }

    fn total(&self) -> usize { self.perm_count * self.combo_count }

    fn build(&self, wl_len: usize, keep_word_order: bool, idx: usize) -> Vec<u16> {
        let perm_idx  = idx / self.combo_count;
        let combo_idx = idx % self.combo_count;

        let m = self.movable_known.len();
        let q = self.movable_missing;

        // qpos: which of the m+q movable *slots* (0-based, indexing into
        // movable_positions) hold a '?'. known_order: the known words, in
        // the order they fill the remaining movable slots.
        let (qpos, known_order): (Vec<usize>, Vec<u16>) = if keep_word_order {
            let qpos: Vec<usize> = self.movable_is_missing.iter().enumerate()
                .filter(|(_, &is_q)| is_q).map(|(i, _)| i).collect();
            (qpos, self.movable_known.clone())
        } else {
            let combo_sel_idx  = perm_idx / self.known_perm_count;
            let known_perm_idx = perm_idx % self.known_perm_count;
            (nth_combination(m + q, q, combo_sel_idx), nth_permutation(&self.movable_known, known_perm_idx))
        };
        let qpos_set: HashSet<usize> = qpos.iter().copied().collect();

        // First q unknown values go to the movable '?' slots (left to
        // right); the rest go to the pinned-missing slots (position order).
        let unknown_vals = nth_missing_combo(wl_len as u16, q + self.pinned_missing.len(), combo_idx);

        let mut seq = vec![0u16; self.total_len];
        for &(pos, w) in &self.pinned_fixed { seq[pos] = w; }

        let (mut ki, mut qi) = (0, 0);
        for (slot_i, &pos) in self.movable_positions.iter().enumerate() {
            if qpos_set.contains(&slot_i) {
                seq[pos] = unknown_vals[qi];
                qi += 1;
            } else {
                seq[pos] = known_order[ki];
                ki += 1;
            }
        }
        for (i, &pos) in self.pinned_missing.iter().enumerate() {
            seq[pos] = unknown_vals[q + i];
        }
        seq
    }
}

// ---------------------------------------------------------------------------
// LazyPhraseIter — O(1) RAM, yields [u16;12] one at a time
// ---------------------------------------------------------------------------

struct LazyPhraseIter {
    /// [slot][alt] = metadata (candidates computed on demand, not stored)
    slot_alts:       Vec<Vec<AltMeta>>,
    slot_orders:     Vec<Vec<usize>>,
    order_pos:       usize,
    alt_idx:         Vec<usize>,
    cand_idx:        Vec<usize>,
    wl_len:          usize,
    keep_word_order: bool,
    first:           bool,
    done:            bool,
}

impl LazyPhraseIter {
    fn new(
        chosen: &[&Slot],
        keep_token_order: bool,
        keep_word_order:  bool,
        wordlist: &'static [&'static str],
    ) -> Self {
        let n      = chosen.len();
        let wl_len = wordlist.len();

        let slot_alts: Vec<Vec<AltMeta>> = chosen.iter().map(|slot| {
            slot.iter().map(|alt| AltMeta::new(alt, wordlist, wl_len, keep_word_order)).collect()
        }).collect();

        let slot_orders: Vec<Vec<usize>> = if keep_token_order || n <= 1 {
            vec![(0..n).collect()]
        } else {
            (0..n).permutations(n).collect()
        };

        let num = slot_alts.len();
        LazyPhraseIter {
            slot_alts,
            slot_orders,
            order_pos: 0,
            alt_idx:   vec![0; num],
            cand_idx:  vec![0; num],
            wl_len,
            keep_word_order,
            first: true,
            done:  num == 0,
        }
    }

    fn build(&self, order: &[usize]) -> Option<[u16; 12]> {
        let mut phrase = [0u16; 12];
        let mut off = 0usize;
        for &si in order {
            let meta  = &self.slot_alts[si][self.alt_idx[si]];
            let words = meta.build(self.wl_len, self.keep_word_order, self.cand_idx[si]);
            if off + words.len() > 12 { return None; }
            phrase[off..off+words.len()].copy_from_slice(&words);
            off += words.len();
        }
        if off == 12 { Some(phrase) } else { None }
    }

    fn advance(&mut self, order: &[usize]) -> bool {
        let n = order.len();
        let mut pos = n as isize - 1;
        while pos >= 0 {
            let si = order[pos as usize];
            self.cand_idx[si] += 1;
            if self.cand_idx[si] < self.slot_alts[si][self.alt_idx[si]].total() { return true; }
            self.cand_idx[si] = 0;
            self.alt_idx[si] += 1;
            if self.alt_idx[si] < self.slot_alts[si].len() { return true; }
            self.alt_idx[si] = 0;
            pos -= 1;
        }
        false
    }
}

impl Iterator for LazyPhraseIter {
    type Item = [u16; 12];

    fn next(&mut self) -> Option<[u16; 12]> {
        if self.done { return None; }
        loop {
            if self.order_pos >= self.slot_orders.len() { self.done = true; return None; }
            let order = self.slot_orders[self.order_pos].clone();

            if self.first {
                self.first = false;
                if let Some(p) = self.build(&order) { return Some(p); }
            }

            if self.advance(&order) {
                if let Some(p) = self.build(&order) { return Some(p); }
                continue;
            }

            // this ordering exhausted — next
            self.order_pos += 1;
            self.first = true;
            for v in &mut self.alt_idx  { *v = 0; }
            for v in &mut self.cand_idx { *v = 0; }
        }
    }
}

// ===========================================================================
// PROGRESS ITERATOR — single overwriting line, seed/s display
// ===========================================================================

/// Best-effort terminal column count (falls back to a conservative default).
/// Progress lines size their bar to this so long lines don't wrap — a
/// wrapped `\r` line returns to the start of the *wrapped* row, not the
/// original one, which is what made progress look like it was scrolling.
#[cfg(target_os = "linux")]
fn terminal_width() -> usize {
    #[repr(C)]
    struct Winsize { row: u16, col: u16, xpixel: u16, ypixel: u16 }
    extern "C" {
        fn ioctl(fd: i32, request: u64, argp: *mut Winsize) -> i32;
    }
    const TIOCGWINSZ: u64 = 0x5413;
    let mut ws = Winsize { row: 0, col: 0, xpixel: 0, ypixel: 0 };
    let ok = unsafe { ioctl(1, TIOCGWINSZ, &mut ws as *mut Winsize) } == 0;
    if ok && ws.col > 0 { ws.col as usize } else { 100 }
}

#[cfg(not(target_os = "linux"))]
fn terminal_width() -> usize { 100 }

struct ProgressIter<I> {
    inner:    I,
    count:    usize,
    total:    Option<usize>,  // None if unknown
    interval: usize,
    start:    Instant,
    last:     Instant,
}

impl<I> ProgressIter<I> {
    fn new(inner: I, total: Option<usize>, interval: usize) -> Self {
        let now = Instant::now();
        ProgressIter { inner, count: 0, total, interval, start: now, last: now }
    }
}

impl<I: Iterator<Item = [u16; 12]>> Iterator for ProgressIter<I> {
    type Item = [u16; 12];

    fn next(&mut self) -> Option<[u16; 12]> {
        let item = self.inner.next()?;
        self.count += 1;

        if self.count % self.interval == 0 {
            let total_s  = self.start.elapsed().as_secs_f64();
            let recent_s = self.last.elapsed().as_secs_f64().max(0.001);
            let avg_tp   = self.count as f64 / total_s.max(0.001);
            let rec_tp   = self.interval as f64 / recent_s;

            let fixed = format!(
                "  {:>10} seeds | {:>6.1}s | avg {:>8}/s | recent {:>8}/s",
                format_number(self.count),
                total_s,
                format_number(avg_tp as usize),
                format_number(rec_tp as usize),
            );

            let eta_str = match self.total {
                Some(tot) if tot > self.count => {
                    let remaining = tot - self.count;
                    let eta_s = remaining as f64 / avg_tp.max(1.0);
                    format!(" | ETA {}", fmt_duration(Duration::from_secs_f64(eta_s)))
                }
                _ => String::new(),
            };
            let pct_str = match self.total {
                Some(tot) if tot > 0 => {
                    format!(" {:.1}%", (self.count as f64 / tot as f64 * 100.0).min(100.0))
                }
                _ => String::new(),
            };

            // Size the bar to whatever room is left after the fixed text, so
            // the whole line stays within the terminal width and never wraps.
            let width    = terminal_width();
            let reserved = fixed.chars().count() + eta_str.chars().count() + pct_str.chars().count() + 4; // " [" + "]" + margin
            let bar_width = width.saturating_sub(reserved).min(30);

            let bar_str = if bar_width > 0 && matches!(self.total, Some(tot) if tot > 0) {
                let pct = (self.count as f64 / self.total.unwrap() as f64).min(1.0);
                let filled = (pct * bar_width as f64) as usize;
                let bar: String = (0..bar_width).map(|i| if i < filled { '█' } else { '░' }).collect();
                format!(" [{}]", bar)
            } else {
                String::new()
            };

            print!("\r{fixed}{bar_str}{pct_str}{eta_str}\x1B[K");
            let _ = io::stdout().flush();
            self.last = Instant::now();
        }
        Some(item)
    }
}

fn fmt_duration(d: Duration) -> String {
    let s = d.as_secs();
    if s >= 3600 { format!("{}h{:02}m{:02}s", s/3600, (s%3600)/60, s%60) }
    else if s >= 60 { format!("{}m{:02}s", s/60, s%60) }
    else { format!("{}s", s) }
}

// ===========================================================================
// BATCH SIZE PROBE
// ===========================================================================

const BYTES_PER_CAND: usize = 28; // d_cand(24) + d_survivors(4)

fn probe_batch_size(
    gpu:      &gpu::Gpu,
    wordlist: &gpu::GpuWordlist,
    h160:     &[u8; 20],
    min_exp:  u32,
    max_exp:  u32,
    pipeline: &str,
) -> usize {
    let start = 1usize.checked_shl(min_exp).unwrap_or(1 << 16);
    let cap   = 1usize.checked_shl(max_exp).unwrap_or(1 << 28);

    fn dummy(n: usize) -> impl Iterator<Item = [u16; 12]> { (0..n).map(|_| [0u16; 12]) }

    let mut batch   = start;
    let mut last_ok = start;

    println!("Probing GPU batch (2^{min_exp}={} .. 2^{max_exp}={}):",
        format_number(start), format_number(cap));

    loop {
        let t  = Instant::now();
        let ok = gpu.search(dummy(batch), wordlist, h160, batch, pipeline);
        let ms = t.elapsed().as_secs_f64() * 1000.0;
        let tp = batch as f64 / t.elapsed().as_secs_f64().max(0.001);

        match ok {
            Ok(_) => {
                println!("  {:>10} → OK  | {:.1} MB | {}/s | {:.0}ms",
                    format_number(batch),
                    (batch * BYTES_PER_CAND) as f64 / (1024.0*1024.0),
                    format_number(tp as usize), ms);
                last_ok = batch;
                let next = batch.saturating_mul(2);
                if next > cap { println!("  → cap 2^{max_exp}={}", format_number(cap)); break; }
                batch = next;
            }
            Err(e) => {
                println!("  {:>10} → FAIL ({e:#}), using {}", format_number(batch), format_number(last_ok));
                break;
            }
        }
    }

    println!("  Batch size : {} ({:.1} MB/batch)\n",
        format_number(last_ok),
        (last_ok * BYTES_PER_CAND) as f64 / (1024.0*1024.0));
    last_ok
}

// ===========================================================================
// TOKENLIST — GPU
// ===========================================================================

fn run_tokenlist_gpu(
    args:     &Args,
    slots:    &[Slot],
    target:   &Target,
    language: Language,
    coin:     Coin,
) -> Result<bool> {
    let gpu_handle  = gpu::Gpu::new()?;
    let wordlist    = language.words_by_prefix("");
    let gpu_wl      = gpu::GpuWordlist::new(wordlist)?;
    let h160        = target_hash20(target)?;
    let pipeline    = pipeline_kernel(coin);
    let target_disp = target.as_display_string();

    let batch_size = args.batch_size.unwrap_or_else(||
        probe_batch_size(&gpu_handle, &gpu_wl, &[0u8;20], args.min_batch, args.max_batch, pipeline));

    let min_tok = args.min_token.unwrap_or(slots.len()).min(slots.len());
    let max_tok = slots.len();
    println!("Using GPU (CUDA) — tokenlist mode — batch {}", format_number(batch_size));
    println!("Slot subsets: {min_tok}..={max_tok}");

    let wall = Instant::now();

    for slot_count in min_tok..=max_tok {
        for chosen_idx in (0..slots.len()).combinations(slot_count) {
            let chosen: Vec<&Slot> = chosen_idx.iter().map(|&i| &slots[i]).collect();

            println!("\nSlot combination {:?}", chosen_idx);
            let _ = io::stdout().flush();

            let iter = ProgressIter::new(
                LazyPhraseIter::new(&chosen, args.keep_token_order, args.keep_word_order, wordlist),
                None, // total unknown without double-iteration
                100_000,
            );

            let t = Instant::now();
            let hit = gpu_handle.search(iter, &gpu_wl, &h160, batch_size, pipeline)?;
            println!(); // newline after \r progress

            let secs = t.elapsed().as_secs_f64();

            match hit {
                Some(h) => {
                    let phrase: Vec<&str> = h.indices.iter().map(|&i| wordlist[i as usize]).collect();
                    println!("Mnemonic : {}", phrase.join(" "));
                    println!("Index    : {}", h.global_index);
                    println!("Address  : {target_disp}");
                    println!("Time     : {:.3}s | Throughput: {}/s",
                        secs,
                        format_number((h.global_index as f64 / secs.max(0.001)) as usize));
                    println!("Wall     : {:.3}s total", wall.elapsed().as_secs_f64());
                    let _ = io::stdout().flush();
                    std::mem::forget(gpu_handle);
                    std::process::exit(0);
                }
                None => println!("  No match ({:.3}s)", secs),
            }
        }
    }

    std::mem::forget(gpu_handle);
    Ok(false)
}

// ===========================================================================
// TOKENLIST — CPU
// ===========================================================================

fn run_tokenlist_cpu(
    args:     &Args,
    slots:    &[Slot],
    target:   &Target,
    language: Language,
    coin:     Coin,
    deriv:    &DerivationPath,
) -> Result<bool> {
    let num_threads = if args.threads == 0 { num_cpus::get() } else { args.threads };
    let _ = rayon::ThreadPoolBuilder::new().num_threads(num_threads).build_global();

    let wordlist    = language.words_by_prefix("");
    let target_str  = target.as_compare_string();
    let target_disp = target.as_display_string();
    let secp = Arc::new(Secp256k1::new());

    let min_tok = args.min_token.unwrap_or(slots.len()).min(slots.len());
    let max_tok = slots.len();
    println!("Using CPU ({num_threads} threads) — tokenlist mode");
    println!("Slot subsets: {min_tok}..={max_tok}");

    let counter      = Arc::new(AtomicUsize::new(0));
    let found        = Arc::new(AtomicBool::new(false));
    let found_phrase = Arc::new(std::sync::Mutex::new(String::new()));
    let found_index  = Arc::new(AtomicUsize::new(0));
    let start        = Instant::now();

    'outer: for slot_count in min_tok..=max_tok {
        if found.load(Ordering::Relaxed) { break; }

        for chosen_idx in (0..slots.len()).combinations(slot_count) {
            if found.load(Ordering::Relaxed) { break 'outer; }

            let chosen: Vec<&Slot> = chosen_idx.iter().map(|&i| &slots[i]).collect();
            println!("\nSlot combination {:?}", chosen_idx);

            let iter = LazyPhraseIter::new(
                &chosen, args.keep_token_order, args.keep_word_order, wordlist,
            );
            let interval = 100_000usize;

            iter.par_bridge().for_each(|phrase_idx| {
                if found.load(Ordering::Relaxed) { return; }

                let i = counter.fetch_add(1, Ordering::Relaxed);
                if i % interval == 0 && i > 0 {
                    let s  = start.elapsed().as_secs_f64();
                    let tp = i as f64 / s.max(0.001);
                    print!("\r  {:>10} seeds | {:>6.1}s | {}/s\x1B[K",
                        format_number(i), s, format_number(tp as usize));
                    let _ = io::stdout().flush();
                }

                let phrase: Vec<&str> = phrase_idx.iter().map(|&idx| wordlist[idx as usize]).collect();
                let phrase_str = phrase.join(" ");

                let mnemonic = match Mnemonic::parse_in_normalized(language, &phrase_str) {
                    Ok(m) => m, Err(_) => return,
                };
                let seed = mnemonic.to_seed("");
                let master = match Xpriv::new_master(Network::Bitcoin, &seed) {
                    Ok(x) => x, Err(_) => return,
                };
                let child = match master.derive_priv(&secp, deriv) {
                    Ok(x) => x, Err(_) => return,
                };
                let addr_str = match coin {
                    Coin::Btc => {
                        let pub_key = PublicKey::new(child.private_key.public_key(&secp));
                        Address::p2pkh(&pub_key, Network::Bitcoin).to_string()
                    }
                    Coin::Eth => eth_address_from_privkey(&child.private_key, &secp),
                };

                if addr_str == target_str {
                    found.store(true, Ordering::SeqCst);
                    found_index.store(i, Ordering::SeqCst);
                    *found_phrase.lock().unwrap() = phrase_str;
                }
            });

            println!();
            if found.load(Ordering::SeqCst) { break 'outer; }
        }
    }

    let secs = start.elapsed().as_secs_f64();
    if found.load(Ordering::SeqCst) {
        let fp  = found_phrase.lock().unwrap();
        let idx = found_index.load(Ordering::SeqCst);
        println!("Mnemonic : {}", *fp);
        println!("Index    : {idx}");
        println!("Address  : {target_disp}");
        println!("Time     : {:.3}s | Throughput: {}/s",
            secs, format_number((idx as f64 / secs.max(0.001)) as usize));
        Ok(true)
    } else {
        Ok(false)
    }
}

// ===========================================================================
// WORD MODE — GPU
// ===========================================================================

fn run_word_gpu(
    args:     &Args,
    target:   &Target,
    language: Language,
    coin:     Coin,
) -> Result<bool> {
    let gpu_handle  = gpu::Gpu::new()?;
    let wordlist    = language.words_by_prefix("");
    let gpu_wl      = gpu::GpuWordlist::new(wordlist)?;
    let h160        = target_hash20(target)?;
    let pipeline    = pipeline_kernel(coin);
    let target_disp = target.as_display_string();

    let batch_size = args.batch_size.unwrap_or_else(||
        probe_batch_size(&gpu_handle, &gpu_wl, &h160, args.min_batch, args.max_batch, pipeline));

    let (owned, pinned_idx) = split_pinned_words(&args.words);

    let mut known_idx: Vec<u16> = Vec::new();
    for w in &owned {
        let pos = wordlist.iter().position(|x| *x == w.as_str())
            .with_context(|| format!("'{w}' not in BIP-39 wordlist"))?;
        known_idx.push(pos as u16);
    }

    let missing = 12 - owned.len();
    let total   = word_mode_total(owned.len(), owned.len() - pinned_idx.len(), wordlist.len(), missing);

    println!("Using GPU (CUDA) — word mode — batch {}", format_number(batch_size));
    if !pinned_idx.is_empty() {
        let pinned_words: Vec<&str> = pinned_idx.iter().map(|&i| owned[i].as_str()).collect();
        println!("Pinned in place (not permuted): {}", pinned_words.join(", "));
    }
    if missing > 0 {
        println!("Completing {missing} missing word(s) from {} BIP-39 words.", wordlist.len());
    }
    println!("Total candidates: {}", format_number(total));

    let cand_iter = candidates::stream(known_idx, pinned_idx.clone(), wordlist.len());
    let prog_iter = ProgressIter::new(cand_iter, Some(total), 100_000);

    let t   = Instant::now();
    let hit = gpu_handle.search(prog_iter, &gpu_wl, &h160, batch_size, pipeline)?;
    println!(); // newline after \r
    let secs = t.elapsed().as_secs_f64();

    match hit {
        Some(h) => {
            let phrase: Vec<&str> = h.indices.iter().map(|&i| wordlist[i as usize]).collect();
            let phrase_str = phrase.join(" ");
            if missing > 0 {
                println!("Recovered : {}", recovered_words(&owned, &phrase_str).join(" "));
            }
            println!("Mnemonic  : {}", phrase_str);
            println!("Index     : {}", h.global_index);
            println!("Address   : {target_disp}");
            println!("Time      : {:.3}s | Throughput: {}/s",
                secs, format_number((h.global_index as f64 / secs.max(0.001)) as usize));
            let _ = io::stdout().flush();
            std::mem::forget(gpu_handle);
            std::process::exit(0);
        }
        None => {
            println!("No match. {:.3}s | {}/s",
                secs, format_number((total as f64 / secs.max(0.001)) as usize));
            std::mem::forget(gpu_handle);
            Ok(false)
        }
    }
}

// ===========================================================================
// WORD MODE — CPU
// ===========================================================================

fn run_word_cpu(
    args:     &Args,
    target:   &Target,
    language: Language,
    coin:     Coin,
    deriv:    &DerivationPath,
) -> Result<bool> {
    let num_threads = if args.threads == 0 { num_cpus::get() } else { args.threads };
    let _ = rayon::ThreadPoolBuilder::new().num_threads(num_threads).build_global();

    let wordlist    = language.words_by_prefix("");
    let target_str  = target.as_compare_string();
    let target_disp = target.as_display_string();
    let secp = Arc::new(Secp256k1::new());

    let (owned, pinned_idx) = split_pinned_words(&args.words);
    let missing = 12 - owned.len();
    let total   = word_mode_total(owned.len(), owned.len() - pinned_idx.len(), wordlist.len(), missing);

    println!("Using CPU ({num_threads} threads) — word mode");
    if !pinned_idx.is_empty() {
        let pinned_words: Vec<&str> = pinned_idx.iter().map(|&i| owned[i].as_str()).collect();
        println!("Pinned in place (not permuted): {}", pinned_words.join(", "));
    }
    if missing > 0 {
        println!("Completing {missing} missing word(s) from {} BIP-39 words.", wordlist.len());
    }
    println!("Total candidates: {}", format_number(total));

    let counter      = Arc::new(AtomicUsize::new(0));
    let found        = Arc::new(AtomicBool::new(false));
    let found_phrase = Arc::new(std::sync::Mutex::new(String::new()));
    let found_index  = Arc::new(AtomicUsize::new(0));
    let start        = Instant::now();

    let candidates = pinned_permutations(&owned, &pinned_idx)
        .flat_map(move |base| insert_missing(base, missing, wordlist).map(|v| v.join(" ")));

    candidates.par_bridge().for_each(|phrase| {
        if found.load(Ordering::Relaxed) { return; }

        let i = counter.fetch_add(1, Ordering::Relaxed);
        if i % 100_000 == 0 && i > 0 {
            let s  = start.elapsed().as_secs_f64();
            let tp = i as f64 / s.max(0.001);
            let eta_s  = (total - i) as f64 / tp.max(1.0);

            let fixed = format!("  {:>10} seeds | {:>6.1}s | {}/s", format_number(i), s, format_number(tp as usize));
            let eta_str = format!(" ETA {}", fmt_duration(Duration::from_secs_f64(eta_s)));
            let pct = (i as f64 / total as f64).min(1.0);
            let pct_str = format!(" {:.1}%", pct * 100.0);

            let width     = terminal_width();
            let reserved  = fixed.chars().count() + eta_str.chars().count() + pct_str.chars().count() + 4;
            let bar_width = width.saturating_sub(reserved).min(25);
            let bar_str = if bar_width > 0 {
                let filled = (pct * bar_width as f64) as usize;
                let bar: String = (0..bar_width).map(|j| if j < filled { '█' } else { '░' }).collect();
                format!(" [{}]", bar)
            } else {
                String::new()
            };

            print!("\r{fixed}{bar_str}{pct_str}{eta_str}\x1B[K");
            let _ = io::stdout().flush();
        }

        let mnemonic = match Mnemonic::parse_in_normalized(language, &phrase) {
            Ok(m) => m, Err(_) => return,
        };
        let seed   = mnemonic.to_seed("");
        let master = match Xpriv::new_master(Network::Bitcoin, &seed) { Ok(x) => x, Err(_) => return };
        let child  = match master.derive_priv(&secp, deriv) { Ok(x) => x, Err(_) => return };
        let addr_str = match coin {
            Coin::Btc => {
                let pub_key = PublicKey::new(child.private_key.public_key(&secp));
                Address::p2pkh(&pub_key, Network::Bitcoin).to_string()
            }
            Coin::Eth => eth_address_from_privkey(&child.private_key, &secp),
        };

        if addr_str == target_str {
            found.store(true, Ordering::SeqCst);
            found_index.store(i, Ordering::SeqCst);
            *found_phrase.lock().unwrap() = phrase;
        }
    });

    println!();
    let secs = start.elapsed().as_secs_f64();

    if found.load(Ordering::SeqCst) {
        let fp  = found_phrase.lock().unwrap();
        let idx = found_index.load(Ordering::SeqCst);
        if missing > 0 {
            println!("Recovered : {}", recovered_words(&owned, &fp).join(" "));
        }
        println!("Mnemonic  : {}", *fp);
        println!("Index     : {idx}");
        println!("Address   : {target_disp}");
        println!("Time      : {:.3}s | Throughput: {}/s",
            secs, format_number((idx as f64 / secs.max(0.001)) as usize));
        Ok(true)
    } else {
        println!("No match. {:.3}s | {}/s",
            secs, format_number((total as f64 / secs.max(0.001)) as usize));
        Ok(false)
    }
}

// ===========================================================================
// HELPERS
// ===========================================================================

fn p2pkh_hash160(addr: &Address<NetworkChecked>) -> Result<[u8; 20]> {
    let spk = addr.script_pubkey();
    let b   = spk.as_bytes();
    if b.len() == 25 && b[0] == 0x76 && b[1] == 0xa9 && b[2] == 0x14 {
        let mut h = [0u8; 20];
        h.copy_from_slice(&b[3..23]);
        Ok(h)
    } else {
        anyhow::bail!("Not a legacy P2PKH address")
    }
}

/// Exact size of the word-mode candidate space actually generated by
/// `pinned_permutations`/`candidates::stream` + `insert_missing`.
///
/// `insert_missing` doesn't drop each missing word into one fixed slot — it
/// tries every gap of the growing sequence, so each of the `missing` words
/// contributes an extra positional factor, not just a `wl_len` value factor.
/// For a known-word count `n` (movable + pinned) and `missing = 12 - n`,
/// that positional factor is `(n+1)(n+2)...(n+missing) == 12!/n!`.
fn word_mode_total(n: usize, movable: usize, wl_len: usize, missing: usize) -> usize {
    let perm_count = factorial(movable);
    let insertion_positions: usize = ((n + 1)..=(n + missing)).product::<usize>().max(1);
    perm_count * insertion_positions * wl_len.pow(missing as u32)
}

fn factorial(n: usize) -> usize {
    (1..=n).product::<usize>().max(1)
}

/// Splits word-mode words into (unquoted words, indices of pinned ones).
/// A word wrapped in literal double quotes (e.g. `"tornado"`) is pinned.
fn split_pinned_words(words: &[String]) -> (Vec<String>, Vec<usize>) {
    let mut out    = Vec::with_capacity(words.len());
    let mut pinned = Vec::new();
    for (i, w) in words.iter().enumerate() {
        if is_pinned_token(w) {
            out.push(strip_pin(w).to_string());
            pinned.push(i);
        } else {
            out.push(w.clone());
        }
    }
    (out, pinned)
}

/// Permutations of `words` that keep the entries at `pinned` indices fixed
/// in place, permuting only the rest. With no pinned indices this is the
/// same as a full permutation of every word.
fn pinned_permutations(
    words:  &[String],
    pinned: &[usize],
) -> Box<dyn Iterator<Item = Vec<String>> + Send> {
    let n = words.len();
    let pinned_set: HashSet<usize> = pinned.iter().copied().collect();
    let pinned_vals: Vec<(usize, String)> = pinned.iter().map(|&i| (i, words[i].clone())).collect();
    let movable: Vec<String> = words.iter().enumerate()
        .filter(|(i, _)| !pinned_set.contains(i))
        .map(|(_, w)| w.clone())
        .collect();
    let movable_len = movable.len();

    Box::new(movable.into_iter().permutations(movable_len).map(move |perm| {
        let mut out = vec![String::new(); n];
        for (i, w) in &pinned_vals { out[*i] = w.clone(); }
        let mut mi = 0;
        for (i, slot) in out.iter_mut().enumerate() {
            if pinned_set.contains(&i) { continue; }
            *slot = perm[mi].clone();
            mi += 1;
        }
        out
    }))
}

fn insert_missing(
    seq: Vec<String>,
    remaining: usize,
    wordlist: &'static [&'static str],
) -> Box<dyn Iterator<Item = Vec<String>> + Send> {
    if remaining == 0 { return Box::new(std::iter::once(seq)); }
    let len = seq.len();
    Box::new((0..=len).flat_map(move |pos| {
        let seq = seq.clone();
        wordlist.iter().flat_map(move |&word| {
            let mut next = Vec::with_capacity(seq.len() + 1);
            next.extend_from_slice(&seq[..pos]);
            next.push(word.to_string());
            next.extend_from_slice(&seq[pos..]);
            insert_missing(next, remaining - 1, wordlist)
        })
    }))
}

fn recovered_words(known: &[String], phrase: &str) -> Vec<String> {
    let mut rem: Vec<String> = known.to_vec();
    let mut out = Vec::new();
    for w in phrase.split_whitespace() {
        if let Some(p) = rem.iter().position(|k| k == w) { rem.remove(p); }
        else { out.push(w.to_string()); }
    }
    out
}

pub fn format_number(n: usize) -> String {
    if n >= 1_000_000_000 { format!("{:.1}G", n as f64 / 1e9) }
    else if n >= 1_000_000 { format!("{:.1}M", n as f64 / 1e6) }
    else if n >= 1_000     { format!("{:.1}K", n as f64 / 1e3) }
    else                   { n.to_string() }
}

fn parse_language(lang: &str) -> Result<Language> {
    match lang.to_lowercase().as_str() {
        "english"             => Ok(Language::English),
        "portuguese"          => Ok(Language::Portuguese),
        "spanish"             => Ok(Language::Spanish),
        "french"              => Ok(Language::French),
        "italian"             => Ok(Language::Italian),
        "czech"               => Ok(Language::Czech),
        "korean"              => Ok(Language::Korean),
        "japanese"            => Ok(Language::Japanese),
        "chinese-simplified"  => Ok(Language::SimplifiedChinese),
        "chinese-traditional" => Ok(Language::TraditionalChinese),
        _ => anyhow::bail!("Unknown language '{lang}'"),
    }
}
