# 🔑 WORD BREAKER GPU WITH TOKENLIST INPUT

> High-performance BIP-39 mnemonic recovery for Bitcoin legacy addresses — with CPU multi-threading and CUDA GPU acceleration.

[![Rust](https://img.shields.io/badge/Rust-1.75%2B-orange?logo=rust)](https://www.rust-lang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![CUDA](https://img.shields.io/badge/CUDA-optional-green?logo=nvidia)](https://developer.nvidia.com/cuda-toolkit)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Tokenlist Format](#tokenlist-format)
- [Examples](#examples)
- [How It Works](#how-it-works)
- [Performance](#performance)
- [Troubleshooting](#troubleshooting)
- [Disclaimer](#disclaimer)

---

## Overview

BTC Mnemonic Recovery Tool helps you recover a forgotten or partially-known BIP-39 mnemonic phrase by exhaustively trying permutations of known word fragments against a target Bitcoin legacy (P2PKH) address.

You provide the words you remember (organized as **slots** and **alternatives**), and the tool finds the exact 12-word phrase that derives your address.

---

## Features

- ✅ **CPU multi-threaded** search via [Rayon](https://github.com/rayon-rs/rayon)
- ✅ **GPU (CUDA) accelerated** search for massive throughput
- ✅ **Flexible tokenlist format** — slots, alternatives, wildcard `?` for missing words
- ✅ **Slot & word permutations** — tries all orderings unless pinned with flags
- ✅ **Subset search** — try combinations of fewer slots via `--min-token`
- ✅ **BIP-39 multi-language** support (10 languages)
- ✅ **Self-test mode** for GPU primitive verification

---

## Requirements

| Component | Requirement |
|-----------|-------------|
| Rust | 1.75 or later |
| CUDA Toolkit | 11.x / 12.x (optional, for GPU mode) |
| GPU | NVIDIA with CUDA support (optional) |
| OS | Linux, macOS, Windows |

---

## Installation

```bash
# Clone the repository
git clone https://github.com/hotan-create/words-breaker-gpu.git
cd word-breaker-gpu

# Build with GPU support (requires CUDA toolkit)
cargo build --release
```

The binary will be at `target/release/word-breaker`.

---

## Usage

```
./target/release/words-breaker 1JKQKyPXm42BPQfu2pevNzT1ej5KBcdaHS --tokenlist tokenlist.txt --min-token 3 --keep-word-order --keep-token-order --max-batch 16
pubkey-mode
./target/release/words-breaker --target-pubkey 0389d9cd176c59c5384044798cc23b650c57255dcd788151f25dfb2d55bb912399 --tokenlist tokenlist.txt --min-token 3 --keep-token-order
```


### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--tokenlist <FILE>` | — | Path to tokenlist file (required) |
| `--language`, `-l` | `english` | BIP-39 wordlist language |
| `--keep-token-order` | off | Do not permute slot order |
| `--keep-word-order` | off | Do not permute word order within slots |
| `--min-token <N>` | all slots | Minimum number of slots to use |
| `--cpu` | off | Force CPU mode even if GPU is available |
| `--selftest` | off | Verify GPU primitives and exit |

---

## Tokenlist Format

The tokenlist file is the core input. Each **line** represents one **slot** (one positional group of words in the mnemonic).

### Rules

```
# Lines starting with '#' are comments and are ignored.
# Blank lines are also ignored.
# Each line = one SLOT
# Alternatives within a line are separated by WHITESPACE
# Words within an alternative are separated by COMMAS
# '?' marks a missing/unknown word to be brute-forced

### Key Concepts

| Concept | Syntax | Meaning |
|---------|--------|---------|
| **Slot** | One line | A positional group in the phrase |
| **Alternative** | Words separated by space | One of multiple possible groups for a slot |
| **Multi-word alternative** | `word1,word2` | Multiple words that appear together |
| **Wildcard** | `?` | Unknown word — searched from full BIP-39 wordlist |

### Example Tokenlist.txt
galaxy,meat,evil,faith
oxygen,donor,?,donkey
popular,friend,oval,venture

note : missing word is fixed order ( you can edit the tokenlist with mutual exclusion
example 

galaxy,meat,evil,faith
?,oxygen,donor,donkey oxygen,?,donor,donkey oxygen,donor,?,donkey oxygen,donor,donkey,?  ##sparate with space every token
popular,friend,oval,venture

``,
```

Total words across chosen slots must equal **12** (standard BIP-39 mnemonic length).

---

## Examples

### Basic CPU search

```bash
./target/release/words-breaker 1At7z8J3t3JJiAqtBTyJuHdCMKx45HmyVp --tokenlist tokenlist.txt --min-token 3 --cpu
```

### GPU search (auto-detect)

./target/release/words-breaker 1At7z8J3t3JJiAqtBTyJuHdCMKx45HmyVp --tokenlist tokenlist.txt --min-token 3


### Fixed slot & word order (no permutations)

```bash
./target/release/words-breaker 1At7z8J3t3JJiAqtBTyJuHdCMKx45HmyVp --tokenlist tokenlist.txt --min-token 3 --keep-token-order
  --keep-word-order
```


### Non-English wordlist

```bash
./target/release/words-breaker 1At7z8J3t3JJiAqtBTyJuHdCMKx45HmyVp --tokenlist tokenlist.txt --min-token 3 --cpu
  --language spanish
```
---

## How It Works

```
Tokenlist File
     │
     ▼
Parse Slots & Alternatives
     │
     ▼
Expand Wildcards (?)  ──────────────────────────────┐
     │                                               │
     ▼                                               │
Generate Permutations                          BIP-39 Wordlist
(slot order + word order)                     (2048 words × lang)
     │
     ▼
Filter: must be exactly 12 words
     │
     ├──── CPU Mode ──── Rayon parallel_iter ─────────┐
     │                                                 │
     └──── GPU Mode ──── CUDA batch kernel ────────────┤
                                                       │
                                                       ▼
                                          BIP-39 → Seed → Master Xpriv
                                          → m/44'/0'/0'/0/0
                                          → P2PKH Address
                                                       │
                                                       ▼
                                            Compare with Target
                                                       │
                                                  Match found!
```

### Derivation Path

All addresses are derived using the standard BIP-44 path:

```
m / 44' / 0' / 0' / 0 / 0
```

---

## Performance

| Mode | Typical Speed | Notes |
|------|--------------|-------|
| CPU (single core) | ~500–2K phrases/sec | Depends on phrase complexity |
| CPU (multi-core) | ~4K–20K phrases/sec | Scales linearly with cores |
| GPU (CUDA) | ~1M+ phrases/sec | Requires NVIDIA GPU with CUDA |

> GPU mode sends the full candidate batch at once per slot combination for maximum throughput.

---

## Supported Languages

| Language | Flag |
|----------|------|
| `english` | 🇬🇧 |
| `spanish` | 🇪🇸 |
| `portuguese` | 🇵🇹 |
| `french` | 🇫🇷 |
| `italian` | 🇮🇹 |
| `czech` | 🇨🇿 |
| `korean` | 🇰🇷 |
| `japanese` | 🇯🇵 |
| `chinese-simplified` | 🇨🇳 |
| `chinese-traditional` | 🇹🇼 |

---

## Troubleshooting

**`GPU unavailable; falling back to CPU`**
CUDA toolkit is not installed or the binary was not built with `--features cuda`. Install the [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) and rebuild.

**`is not in the BIP-39 wordlist`**
One of the words in your tokenlist is not a valid BIP-39 word. Check spelling — all words must match the official BIP-39 wordlist exactly.

**`Only mainnet legacy addresses are supported`**
The tool only supports mainnet P2PKH addresses (starting with `1`). Segwit (`3...`, `bc1...`) addresses are not supported.

**`Total word count must equal 12`**
The combined word count across all chosen slots must be exactly 12. Adjust your tokenlist or use `--min-token` to select fewer slots.

---

## Disclaimer

> ⚠️ **This tool is intended solely for recovering access to your own Bitcoin wallet.**
> Using it to attempt unauthorized access to wallets you do not own is illegal and unethical.
> The authors accept no responsibility for misuse.

---

## License

MIT © 2024
