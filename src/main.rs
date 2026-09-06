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
    /// • '?' = unknown word, brute-forced from full BIP-39 list.
    /// • "word" (in literal double quotes) = pinned: kept at its position,
    ///   excluded from permutation, even without --keep-word-order.
    /// • A line may start with an integer, e.g. "4 fiber" — this fixes the
    ///   slot at absolute position 4 (1-indexed) of the final phrase. Only
    ///   effective when --keep-token-order is NOT set (it already fixes
    ///   every slot's position). A slot with a fixed position must be a
    ///   single word/'?' per alternative (no multi-word alternatives).
    ///   Slots without a leading number keep permuting freely across
    ///   whatever positions are left.
    ///
    /// EXAMPLE
    ///   zebra,"tornado",gravity,?   abandon,art   <- slot 1 (2 alternatives)
    ///   orbit,galaxy                               <- slot 2
    ///   venture,sun                                <- slot 3
    /// (in alt 1: tornado stays 2nd; zebra/gravity permute; last word brute-forced)
    ///
    /// EXAMPLE (fixed positions, rest permutes freely)
    ///   1 dutch          <- word 1 of the phrase is fixed to "dutch"
    ///   4 fiber          <- word 4 is fixed to "fiber"
    ///   ?                <- an unpinned slot, brute-forced, goes anywhere left
    ///   fork sponsor     <- an unpinned slot: alt "fork" or alt "sponsor"
    ///
    /// Total words across chosen slots must equal 12.
    #[arg(long, value_name = "FILE")]
    tokenlist: Option<PathBuf>,

    /// Keep slot order as written (no slot permutations). This also fixes
    /// every slot's absolute position to its line order, so any explicit
    /// position numbers in the tokenlist must already match that order.
    #[arg(long)]
    keep_token_order: bool,

    /// Keep word order within each slot (tokenlist mode; no intra-slot
    /// permutations). Not used in word mode — there, pin individual words
    /// with literal double quotes instead (see the positional args above).
    #[arg(long)]
    keep_word_order: bool,

    /// Minimum number of slots to use (default: all). Slots with an
    /// explicit fixed position (see --tokenlist) always count toward this
    /// and are always included; this only trims down the unpositioned ones.
    #[arg(long, alias = "min-tokens", value_name = "N")]
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
    Missing,
}

type Alternative = Vec<Token>;
type Slot        = Vec<Alternative>;

/// One parsed tokenlist line: its alternatives, plus an optional explicit
/// absolute position (1-indexed) in the final 12-word phrase, set by a
/// leading integer on the line (e.g. `4 fiber`).
#[derive(Debug, Clone)]
struct TokenlistSlot {
    position: Option<usize>,
    alts:     Slot,
}

/// A word wrapped in literal double quotes (`"..."`, length >= 2) marks it
/// pinned. Shared by tokenlist parsing and word-mode CLI args.
fn is_pinned_token(s: &str) -> bool {
    s.len() >= 2 && s.starts_with('"') && s.ends_with('"')
}

fn strip_pin(s: &str) -> &str {
    if is_pinned_token(s) { &s[1..s.len() - 1] } else { s }
}

/// Splits slot indices into (fixed, free) by whether they carry an
/// explicit position. When --keep-token-order is set, line order already
/// fixes every slot's position, so all slots are treated as "free" here
/// (in file order) rather than double-handled as fixed.
fn split_fixed_free(slots: &[TokenlistSlot], keep_token_order: bool) -> (Vec<usize>, Vec<usize>) {
    if keep_token_order {
        return (Vec::new(), (0..slots.len()).collect());
    }
    let fixed: Vec<usize> = slots.iter().enumerate()
        .filter(|(_, s)| s.position.is_some()).map(|(i, _)| i).collect();
    let free: Vec<usize> = slots.iter().enumerate()
        .filter(|(_, s)| s.position.is_none()).map(|(i, _)| i).collect();
    (fixed, free)
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
        validate_slots(&slots, language, args.keep_token_order)?;

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

fn parse_tokenlist(path: &PathBuf) -> Result<Vec<TokenlistSlot>> {
    let content = fs::read_to_string(path)
        .with_context(|| format!("Cannot read tokenlist: {}", path.display()))?;

    let mut slots: Vec<TokenlistSlot> = Vec::new();
    for (lineno, raw) in content.lines().enumerate() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') { continue; }

        // A leading bare integer fixes this slot's absolute position
        // (1-indexed) in the final phrase, e.g. "4 fiber". Consume it
        // before splitting the rest into alternatives as usual.
        let mut parts = line.split_whitespace().peekable();
        let mut position: Option<usize> = None;
        if let Some(&first) = parts.peek() {
            if let Ok(n) = first.parse::<usize>() {
                position = Some(n);
                parts.next();
            }
        }
        let rest: Vec<&str> = parts.collect();
        anyhow::ensure!(
            !rest.is_empty(),
            "Line {}: has a position number but no word/alternative after it",
            lineno + 1
        );

        let alternatives: Vec<Alternative> = rest
            .into_iter()
            .map(|alt_str| {
                alt_str.split(',')
                    .filter(|t| !t.is_empty())
                    .map(|t| {
                        let t = t.trim();
                        if t == "?" {
                            Token::Missing
                        } else if is_pinned_token(t) {
                            Token::PinnedWord(strip_pin(t).to_string())
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
        slots.push(TokenlistSlot { position, alts: alternatives });
    }

    anyhow::ensure!(!slots.is_empty(), "Tokenlist is empty or has no valid lines");
    let positioned = slots.iter().filter(|s| s.position.is_some()).count();
    if positioned > 0 {
        println!(
            "Loaded {} slot(s) from tokenlist ({} with a fixed position).",
            slots.len(), positioned
        );
    } else {
        println!("Loaded {} slot(s) from tokenlist.", slots.len());
    }
    Ok(slots)
}

fn validate_slots(slots: &[TokenlistSlot], language: Language, keep_token_order: bool) -> Result<()> {
    let wl: &'static [&'static str] = language.words_by_prefix("");
    let mut seen_positions: HashSet<usize> = HashSet::new();

    for (si, slot) in slots.iter().enumerate() {
        for (ai, alt) in slot.alts.iter().enumerate() {
            for tok in alt {
                if let Token::Word(w) | Token::PinnedWord(w) = tok {
                    anyhow::ensure!(
                        wl.contains(&w.as_str()),
                        "Slot {}, alt {}: '{}' not in BIP-39 wordlist", si+1, ai+1, w
                    );
                }
            }
        }

        if let Some(pos) = slot.position {
            anyhow::ensure!(
                (1..=12).contains(&pos),
                "Slot {}: fixed position {} out of range (must be 1..=12)", si+1, pos
            );
            anyhow::ensure!(
                seen_positions.insert(pos),
                "Slot {}: fixed position {} is already used by another slot", si+1, pos
            );
            for (ai, alt) in slot.alts.iter().enumerate() {
                anyhow::ensure!(
                    alt.len() == 1,
                    "Slot {} (fixed position {}), alt {}: must be exactly one word/'?' \
                     — a slot with a fixed position occupies a single spot in the phrase",
                    si+1, pos, ai+1
                );
            }
            if keep_token_order {
                anyhow::ensure!(
                    pos == si + 1,
                    "Slot {}: fixed position {} conflicts with --keep-token-order, \
                     which already fixes this slot at position {} (its line order)",
                    si+1, pos, si+1
                );
            }
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Expand one alternative → (known_word_indices, missing_positions)
// ---------------------------------------------------------------------------

fn expand_alternative(
    alt: &Alternative,
    wordlist: &'static [&'static str],
) -> (Vec<u16>, Vec<(usize, u16)>, Vec<usize>) {
    let mut movable: Vec<u16>       = Vec::new();
    let mut pinned:  Vec<(usize, u16)> = Vec::new();
    let mut miss:    Vec<usize>     = Vec::new();
    for (i, tok) in alt.iter().enumerate() {
        match tok {
            Token::Word(w)       => movable.push(wordlist.iter().position(|x| *x == w.as_str()).unwrap() as u16),
            Token::PinnedWord(w) => pinned.push((i, wordlist.iter().position(|x| *x == w.as_str()).unwrap() as u16)),
            Token::Missing       => miss.push(i),
        }
    }
    (movable, pinned, miss)
}

// ---------------------------------------------------------------------------
// slot_candidates: expand one alternative into Vec<Vec<u16>> of partial phrases
// ---------------------------------------------------------------------------

fn slot_candidates(
    movable: &[u16],
    pinned:  &[(usize, u16)],
    miss:    &[usize],
    total_len: usize,
    keep_word_order: bool,
    wl_len: usize,
) -> Vec<Vec<u16>> {
    let movable_perms: Vec<Vec<u16>> = if keep_word_order {
        vec![movable.to_vec()]
    } else {
        movable.iter().copied().permutations(movable.len()).collect()
    };

    let mut out = Vec::new();
    for kp in &movable_perms {
        for mv in missing_combos(wl_len as u16, miss.len()) {
            let mut seq = vec![0u16; total_len];
            for &(pos, w) in pinned { seq[pos] = w; }
            let (mut ki, mut mi) = (0, 0);
            for pos in 0..total_len {
                if pinned.iter().any(|&(p, _)| p == pos) { continue; }
                if miss.contains(&pos) { seq[pos] = mv[mi]; mi += 1; }
                else                   { seq[pos] = kp[ki]; ki += 1; }
            }
            out.push(seq);
        }
    }
    out
}

/// Yield all n^k combinations with replacement (k indices from 0..n).
fn missing_combos(n: u16, k: usize) -> Vec<Vec<u16>> {
    if k == 0 { return vec![vec![]]; }
    let mut result = Vec::new();
    let sub = missing_combos(n, k - 1);
    for i in 0..n {
        for mut s in sub.clone() {
            s.insert(0, i);
            result.push(s);
        }
    }
    result
}

// ---------------------------------------------------------------------------
// LazyPhraseIter — O(1) RAM, yields [u16;12] one at a time
// ---------------------------------------------------------------------------

struct LazyPhraseIter {
    /// [slot][alt][cand_idx] = partial Vec<u16>. Fixed-position slots occupy
    /// indices `0..fixed_count`; the remaining slots (free — no fixed
    /// position) occupy `fixed_count..`.
    slot_alts:  Vec<Vec<Vec<Vec<u16>>>>,
    fixed_count: usize,
    /// 0-indexed absolute phrase position for each fixed slot, same order
    /// as `slot_alts[0..fixed_count]`.
    fixed_positions: Vec<usize>,
    /// Sorted 0-indexed phrase positions not claimed by any fixed slot;
    /// free slots fill these, in order, as picked by `slot_orders`.
    remaining_positions: Vec<usize>,
    /// Permutations of free-slot indices (0-indexed within the free group).
    slot_orders: Vec<Vec<usize>>,
    /// Precomputed per-order advance chains: `(0..fixed_count)` followed by
    /// `fixed_count + slot_orders[k][..]`, one entry per `slot_orders[k]`.
    /// Computed once in `new()` so `advance()`/`next()` never need to
    /// clone/rebuild this on the hot per-candidate path.
    chains:      Vec<Vec<usize>>,
    order_pos:   usize,
    alt_idx:     Vec<usize>,
    cand_idx:    Vec<usize>,
    first:       bool,
    done:        bool,
}

impl LazyPhraseIter {
    /// `fixed` slots each carry an explicit absolute position (validated to
    /// be a single word/'?' per alternative) and are never reordered.
    /// `free` slots have no fixed position and permute across whatever
    /// positions `fixed` doesn't claim, unless `keep_token_order` is set.
    fn new(
        fixed: &[&TokenlistSlot],
        free:  &[&TokenlistSlot],
        keep_token_order: bool,
        keep_word_order:  bool,
        wordlist: &'static [&'static str],
    ) -> Self {
        let wl_len      = wordlist.len();
        let fixed_count = fixed.len();
        let free_count  = free.len();

        let chosen_alts: Vec<&Slot> = fixed.iter().map(|s| &s.alts)
            .chain(free.iter().map(|s| &s.alts))
            .collect();

        let slot_alts: Vec<Vec<Vec<Vec<u16>>>> = chosen_alts.iter().map(|slot| {
            slot.iter().map(|alt| {
                let (movable, pinned, miss) = expand_alternative(alt, wordlist);
                slot_candidates(&movable, &pinned, &miss, alt.len(), keep_word_order, wl_len)
            }).collect()
        }).collect();

        let fixed_positions: Vec<usize> = fixed.iter()
            .map(|s| s.position.expect("fixed slot must have a position") - 1)
            .collect();
        let mut used = [false; 12];
        for &p in &fixed_positions { used[p] = true; }
        let remaining_positions: Vec<usize> = (0..12).filter(|&i| !used[i]).collect();

        let slot_orders: Vec<Vec<usize>> = if keep_token_order || free_count <= 1 {
            vec![(0..free_count).collect()]
        } else {
            (0..free_count).permutations(free_count).collect()
        };

        // Precompute the advance-chain for each order once, up front —
        // this used to be rebuilt (with a heap allocation) on every single
        // `next()` call, which throttled candidate throughput badly.
        let chains: Vec<Vec<usize>> = slot_orders.iter().map(|order| {
            let mut c: Vec<usize> = (0..fixed_count).collect();
            c.extend(order.iter().map(|&fsi| fixed_count + fsi));
            c
        }).collect();

        let total = fixed_count + free_count;
        LazyPhraseIter {
            slot_alts,
            fixed_count,
            fixed_positions,
            remaining_positions,
            slot_orders,
            chains,
            order_pos: 0,
            alt_idx:   vec![0; total],
            cand_idx:  vec![0; total],
            first: true,
            done:  total == 0,
        }
    }

    fn build(&self) -> Option<[u16; 12]> {
        let mut phrase = [0u16; 12];

        for fi in 0..self.fixed_count {
            let words = &self.slot_alts[fi][self.alt_idx[fi]][self.cand_idx[fi]];
            if words.len() != 1 { return None; }
            phrase[self.fixed_positions[fi]] = words[0];
        }

        let order = &self.slot_orders[self.order_pos];
        let mut off = 0usize;
        for &fsi in order {
            let si = self.fixed_count + fsi;
            let words = &self.slot_alts[si][self.alt_idx[si]][self.cand_idx[si]];
            if off + words.len() > self.remaining_positions.len() { return None; }
            for (k, &w) in words.iter().enumerate() {
                phrase[self.remaining_positions[off + k]] = w;
            }
            off += words.len();
        }
        if off == self.remaining_positions.len() { Some(phrase) } else { None }
    }

    /// Advances candidate/alt indices along the chain for the current
    /// `order_pos`. Reads `self.chains[self.order_pos]` (precomputed in
    /// `new()`) index-by-index — no allocation on this hot path.
    fn advance(&mut self) -> bool {
        let n = self.chains[self.order_pos].len();
        let mut pos = n as isize - 1;
        while pos >= 0 {
            let si = self.chains[self.order_pos][pos as usize];
            self.cand_idx[si] += 1;
            if self.cand_idx[si] < self.slot_alts[si][self.alt_idx[si]].len() { return true; }
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

            if self.first {
                self.first = false;
                if let Some(p) = self.build() { return Some(p); }
            }

            if self.advance() {
                if let Some(p) = self.build() { return Some(p); }
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
    slots:    &[TokenlistSlot],
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

    let (fixed_all, free_all) = split_fixed_free(slots, args.keep_token_order);
    let fixed_count = fixed_all.len();

    let min_tok = args.min_token.unwrap_or(slots.len()).min(slots.len());
    let max_tok = slots.len();
    let min_free = min_tok.saturating_sub(fixed_count);
    let max_free = max_tok.saturating_sub(fixed_count).min(free_all.len());

    println!("Using GPU (CUDA) — tokenlist mode — batch {}", format_number(batch_size));
    println!("Slot subsets: {min_tok}..={max_tok}");
    if fixed_count > 0 {
        println!("Fixed-position slots: {fixed_count} (always included, never reordered)");
    }

    let wall = Instant::now();

    for free_count in min_free..=max_free {
        for free_chosen in free_all.iter().copied().combinations(free_count) {
            let fixed_slots: Vec<&TokenlistSlot> = fixed_all.iter().map(|&i| &slots[i]).collect();
            let free_slots:  Vec<&TokenlistSlot> = free_chosen.iter().map(|&i| &slots[i]).collect();
            let chosen_idx: Vec<usize> = fixed_all.iter().copied().chain(free_chosen.iter().copied()).collect();

            println!("\nSlot combination {:?}", chosen_idx);
            let _ = io::stdout().flush();

            let iter = ProgressIter::new(
                LazyPhraseIter::new(&fixed_slots, &free_slots, args.keep_token_order, args.keep_word_order, wordlist),
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
    slots:    &[TokenlistSlot],
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

    let (fixed_all, free_all) = split_fixed_free(slots, args.keep_token_order);
    let fixed_count = fixed_all.len();

    let min_tok = args.min_token.unwrap_or(slots.len()).min(slots.len());
    let max_tok = slots.len();
    let min_free = min_tok.saturating_sub(fixed_count);
    let max_free = max_tok.saturating_sub(fixed_count).min(free_all.len());

    println!("Using CPU ({num_threads} threads) — tokenlist mode");
    println!("Slot subsets: {min_tok}..={max_tok}");
    if fixed_count > 0 {
        println!("Fixed-position slots: {fixed_count} (always included, never reordered)");
    }

    let counter      = Arc::new(AtomicUsize::new(0));
    let found        = Arc::new(AtomicBool::new(false));
    let found_phrase = Arc::new(std::sync::Mutex::new(String::new()));
    let found_index  = Arc::new(AtomicUsize::new(0));
    let start        = Instant::now();

    'outer: for free_count in min_free..=max_free {
        if found.load(Ordering::Relaxed) { break; }

        for free_chosen in free_all.iter().copied().combinations(free_count) {
            if found.load(Ordering::Relaxed) { break 'outer; }

            let fixed_slots: Vec<&TokenlistSlot> = fixed_all.iter().map(|&i| &slots[i]).collect();
            let free_slots:  Vec<&TokenlistSlot> = free_chosen.iter().map(|&i| &slots[i]).collect();
            let chosen_idx: Vec<usize> = fixed_all.iter().copied().chain(free_chosen.iter().copied()).collect();
            println!("\nSlot combination {:?}", chosen_idx);

            let iter = LazyPhraseIter::new(
                &fixed_slots, &free_slots, args.keep_token_order, args.keep_word_order, wordlist,
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
