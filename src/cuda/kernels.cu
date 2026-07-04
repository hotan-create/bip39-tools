// CUDA device implementation of the words-breaker hot path.
//
// Each primitive was built and verified against the CPU crates via `--selftest`
// (see src/gpu.rs): SHA-256, SHA-512, RIPEMD-160, HMAC-SHA512, PBKDF2, secp256k1
// (priv->pubkey and scalar add mod n), and BIP32 m/44'/0'/0'/0/0 seed->hash160.
// The full search is split into k_filter (cheap BIP-39 checksum, compacting
// survivors) and k_pipeline (heavy derivation), so the heavy pass has no warp
// divergence.
//
// All multi-byte values follow the relevant standard's byte order (big-endian
// for SHA, little-endian for RIPEMD-160), independent of GPU endianness, so the
// code is endianness-correct by construction.

#include <stdint.h>

typedef uint8_t  u8;
typedef uint32_t u32;
typedef uint64_t u64;

__device__ __forceinline__ u32 rotr32(u32 x, u32 n) { return (x >> n) | (x << (32 - n)); }
__device__ __forceinline__ u32 rotl32(u32 x, u32 n) { return (x << n) | (x >> (32 - n)); }
__device__ __forceinline__ u64 rotr64(u64 x, u32 n) { return (x >> n) | (x << (64 - n)); }

__device__ __forceinline__ void dmemcpy(u8* dst, const u8* src, u32 n) {
    for (u32 i = 0; i < n; i++) dst[i] = src[i];
}

// ===========================================================================
// SHA-256 (FIPS 180-4) — streaming
// ===========================================================================

__constant__ u32 K256[64] = {
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
    0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2,
};

typedef struct { u32 h[8]; u64 total; u8 buf[64]; u32 n; } sha256_ctx;

__device__ void sha256_init(sha256_ctx* c) {
    c->h[0]=0x6a09e667; c->h[1]=0xbb67ae85; c->h[2]=0x3c6ef372; c->h[3]=0xa54ff53a;
    c->h[4]=0x510e527f; c->h[5]=0x9b05688c; c->h[6]=0x1f83d9ab; c->h[7]=0x5be0cd19;
    c->total = 0; c->n = 0;
}

__device__ void sha256_transform(u32 h[8], const u8 block[64]) {
    // 16-word rolling message schedule: w[(i)&15] holds w[i-16] until overwritten,
    // so only 16 words are kept live (vs 64), cutting per-thread register/local
    // memory pressure. With #pragma unroll the (i&15) indices fold to constants.
    u32 w[16];
    #pragma unroll
    for (int i = 0; i < 16; i++)
        w[i] = ((u32)block[i*4]<<24)|((u32)block[i*4+1]<<16)|((u32)block[i*4+2]<<8)|((u32)block[i*4+3]);
    u32 a=h[0],b=h[1],c=h[2],d=h[3],e=h[4],f=h[5],g=h[6],hh=h[7];
    #pragma unroll
    for (int i = 0; i < 64; i++) {
        u32 wi;
        if (i < 16) {
            wi = w[i & 15];
        } else {
            u32 w15 = w[(i+1) & 15];   // w[i-15]
            u32 w2  = w[(i+14) & 15];  // w[i-2]
            u32 s0 = rotr32(w15,7) ^ rotr32(w15,18) ^ (w15>>3);
            u32 s1 = rotr32(w2,17) ^ rotr32(w2,19) ^ (w2>>10);
            wi = w[i & 15] + s0 + w[(i+9) & 15] + s1; // w[i-16] + s0 + w[i-7] + s1
            w[i & 15] = wi;
        }
        u32 S1 = rotr32(e,6) ^ rotr32(e,11) ^ rotr32(e,25);
        u32 ch = (e & f) ^ ((~e) & g);
        u32 t1 = hh + S1 + ch + K256[i] + wi;
        u32 S0 = rotr32(a,2) ^ rotr32(a,13) ^ rotr32(a,22);
        u32 maj = (a & b) ^ (a & c) ^ (b & c);
        u32 t2 = S0 + maj;
        hh=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
    }
    h[0]+=a; h[1]+=b; h[2]+=c; h[3]+=d; h[4]+=e; h[5]+=f; h[6]+=g; h[7]+=hh;
}

__device__ void sha256_update(sha256_ctx* c, const u8* data, u32 len) {
    c->total += len;
    while (len) {
        u32 take = (64 - c->n < len) ? (64 - c->n) : len;
        dmemcpy(c->buf + c->n, data, take);
        c->n += take; data += take; len -= take;
        if (c->n == 64) { sha256_transform(c->h, c->buf); c->n = 0; }
    }
}

__device__ void sha256_final(sha256_ctx* c, u8 out[32]) {
    // Pad directly in the context buffer (one or two block flushes) instead of
    // streaming the padding one byte at a time through sha256_update.
    u64 bits = c->total * 8;
    u32 n = c->n;
    c->buf[n++] = 0x80;
    if (n > 56) {
        while (n < 64) c->buf[n++] = 0;
        sha256_transform(c->h, c->buf);
        n = 0;
    }
    while (n < 56) c->buf[n++] = 0;
    for (int i = 0; i < 8; i++) c->buf[56 + i] = (u8)(bits >> (56 - i*8)); // big-endian
    sha256_transform(c->h, c->buf);
    for (int i = 0; i < 8; i++) {
        out[i*4]   = (u8)(c->h[i] >> 24);
        out[i*4+1] = (u8)(c->h[i] >> 16);
        out[i*4+2] = (u8)(c->h[i] >> 8);
        out[i*4+3] = (u8)(c->h[i]);
    }
}

__device__ void sha256(const u8* data, u32 len, u8 out[32]) {
    sha256_ctx c; sha256_init(&c); sha256_update(&c, data, len); sha256_final(&c, out);
}

// ===========================================================================
// SHA-512 (FIPS 180-4) — streaming
// ===========================================================================

__constant__ u64 K512[80] = {
    0x428a2f98d728ae22ULL,0x7137449123ef65cdULL,0xb5c0fbcfec4d3b2fULL,0xe9b5dba58189dbbcULL,
    0x3956c25bf348b538ULL,0x59f111f1b605d019ULL,0x923f82a4af194f9bULL,0xab1c5ed5da6d8118ULL,
    0xd807aa98a3030242ULL,0x12835b0145706fbeULL,0x243185be4ee4b28cULL,0x550c7dc3d5ffb4e2ULL,
    0x72be5d74f27b896fULL,0x80deb1fe3b1696b1ULL,0x9bdc06a725c71235ULL,0xc19bf174cf692694ULL,
    0xe49b69c19ef14ad2ULL,0xefbe4786384f25e3ULL,0x0fc19dc68b8cd5b5ULL,0x240ca1cc77ac9c65ULL,
    0x2de92c6f592b0275ULL,0x4a7484aa6ea6e483ULL,0x5cb0a9dcbd41fbd4ULL,0x76f988da831153b5ULL,
    0x983e5152ee66dfabULL,0xa831c66d2db43210ULL,0xb00327c898fb213fULL,0xbf597fc7beef0ee4ULL,
    0xc6e00bf33da88fc2ULL,0xd5a79147930aa725ULL,0x06ca6351e003826fULL,0x142929670a0e6e70ULL,
    0x27b70a8546d22ffcULL,0x2e1b21385c26c926ULL,0x4d2c6dfc5ac42aedULL,0x53380d139d95b3dfULL,
    0x650a73548baf63deULL,0x766a0abb3c77b2a8ULL,0x81c2c92e47edaee6ULL,0x92722c851482353bULL,
    0xa2bfe8a14cf10364ULL,0xa81a664bbc423001ULL,0xc24b8b70d0f89791ULL,0xc76c51a30654be30ULL,
    0xd192e819d6ef5218ULL,0xd69906245565a910ULL,0xf40e35855771202aULL,0x106aa07032bbd1b8ULL,
    0x19a4c116b8d2d0c8ULL,0x1e376c085141ab53ULL,0x2748774cdf8eeb99ULL,0x34b0bcb5e19b48a8ULL,
    0x391c0cb3c5c95a63ULL,0x4ed8aa4ae3418acbULL,0x5b9cca4f7763e373ULL,0x682e6ff3d6b2b8a3ULL,
    0x748f82ee5defb2fcULL,0x78a5636f43172f60ULL,0x84c87814a1f0ab72ULL,0x8cc702081a6439ecULL,
    0x90befffa23631e28ULL,0xa4506cebde82bde9ULL,0xbef9a3f7b2c67915ULL,0xc67178f2e372532bULL,
    0xca273eceea26619cULL,0xd186b8c721c0c207ULL,0xeada7dd6cde0eb1eULL,0xf57d4f7fee6ed178ULL,
    0x06f067aa72176fbaULL,0x0a637dc5a2c898a6ULL,0x113f9804bef90daeULL,0x1b710b35131c471bULL,
    0x28db77f523047d84ULL,0x32caab7b40c72493ULL,0x3c9ebe0a15c9bebcULL,0x431d67c49c100d4cULL,
    0x4cc5d4becb3e42b6ULL,0x597f299cfc657e2aULL,0x5fcb6fab3ad6faecULL,0x6c44198c4a475817ULL,
};

typedef struct { u64 h[8]; u64 total; u8 buf[128]; u32 n; } sha512_ctx;

__device__ void sha512_init(sha512_ctx* c) {
    c->h[0]=0x6a09e667f3bcc908ULL; c->h[1]=0xbb67ae8584caa73bULL;
    c->h[2]=0x3c6ef372fe94f82bULL; c->h[3]=0xa54ff53a5f1d36f1ULL;
    c->h[4]=0x510e527fade682d1ULL; c->h[5]=0x9b05688c2b3e6c1fULL;
    c->h[6]=0x1f83d9abfb41bd6bULL; c->h[7]=0x5be0cd19137e2179ULL;
    c->total = 0; c->n = 0;
}

__device__ void sha512_transform(u64 h[8], const u8 block[128]) {
    // 16-word rolling message schedule (see sha256_transform): keeps only 16 of
    // the 80 schedule words live, which is a big register/local-memory win for
    // the 64-bit state — and SHA-512 is the hot primitive (PBKDF2 runs it 4096x
    // per candidate).
    u64 w[16];
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        w[i] = ((u64)block[i*8]<<56)|((u64)block[i*8+1]<<48)|((u64)block[i*8+2]<<40)|((u64)block[i*8+3]<<32)
             |((u64)block[i*8+4]<<24)|((u64)block[i*8+5]<<16)|((u64)block[i*8+6]<<8)|((u64)block[i*8+7]);
    }
    u64 a=h[0],b=h[1],c=h[2],d=h[3],e=h[4],f=h[5],g=h[6],hh=h[7];
    #pragma unroll
    for (int i = 0; i < 80; i++) {
        u64 wi;
        if (i < 16) {
            wi = w[i & 15];
        } else {
            u64 w15 = w[(i+1) & 15];   // w[i-15]
            u64 w2  = w[(i+14) & 15];  // w[i-2]
            u64 s0 = rotr64(w15,1) ^ rotr64(w15,8) ^ (w15>>7);
            u64 s1 = rotr64(w2,19) ^ rotr64(w2,61) ^ (w2>>6);
            wi = w[i & 15] + s0 + w[(i+9) & 15] + s1; // w[i-16] + s0 + w[i-7] + s1
            w[i & 15] = wi;
        }
        u64 S1 = rotr64(e,14) ^ rotr64(e,18) ^ rotr64(e,41);
        u64 ch = (e & f) ^ ((~e) & g);
        u64 t1 = hh + S1 + ch + K512[i] + wi;
        u64 S0 = rotr64(a,28) ^ rotr64(a,34) ^ rotr64(a,39);
        u64 maj = (a & b) ^ (a & c) ^ (b & c);
        u64 t2 = S0 + maj;
        hh=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
    }
    h[0]+=a; h[1]+=b; h[2]+=c; h[3]+=d; h[4]+=e; h[5]+=f; h[6]+=g; h[7]+=hh;
}

__device__ void sha512_update(sha512_ctx* c, const u8* data, u32 len) {
    c->total += len;
    while (len) {
        u32 take = (128 - c->n < len) ? (128 - c->n) : len;
        dmemcpy(c->buf + c->n, data, take);
        c->n += take; data += take; len -= take;
        if (c->n == 128) { sha512_transform(c->h, c->buf); c->n = 0; }
    }
}

__device__ void sha512_final(sha512_ctx* c, u8 out[64]) {
    // We only ever hash messages well under 2^64 bytes, so the high 64 bits of
    // the 128-bit length field are always zero.
    // Pad directly in the context buffer instead of streaming padding byte-by-byte.
    u64 bits = c->total * 8;
    u32 n = c->n;
    c->buf[n++] = 0x80;
    if (n > 112) {
        while (n < 128) c->buf[n++] = 0;
        sha512_transform(c->h, c->buf);
        n = 0;
    }
    while (n < 112) c->buf[n++] = 0;
    // 128-bit length: high 64 bits are always zero (messages are tiny), low 64
    // bits big-endian.
    for (int i = 0; i < 8; i++) c->buf[112 + i] = 0;
    for (int i = 0; i < 8; i++) c->buf[120 + i] = (u8)(bits >> (56 - i*8));
    sha512_transform(c->h, c->buf);
    for (int i = 0; i < 8; i++)
        for (int j = 0; j < 8; j++) out[i*8+j] = (u8)(c->h[i] >> (56 - j*8));
}

__device__ void sha512(const u8* data, u32 len, u8 out[64]) {
    sha512_ctx c; sha512_init(&c); sha512_update(&c, data, len); sha512_final(&c, out);
}

// ===========================================================================
// RIPEMD-160 (one-shot; inputs here are always small — a 32-byte SHA-256 digest)
// ===========================================================================

__device__ __forceinline__ u32 rmd_f(int j, u32 x, u32 y, u32 z) {
    if (j < 16) return x ^ y ^ z;
    if (j < 32) return (x & y) | (~x & z);
    if (j < 48) return (x | ~y) ^ z;
    if (j < 64) return (x & z) | (y & ~z);
    return x ^ (y | ~z);
}

__device__ void ripemd160(const u8* msg, u32 len, u8 out[20]) {
    static const u32 KL[5] = {0x00000000,0x5a827999,0x6ed9eba1,0x8f1bbcdc,0xa953fd4e};
    static const u32 KR[5] = {0x50a28be6,0x5c4dd124,0x6d703ef3,0x7a6d76e9,0x00000000};
    static const u8 RL[80] = {
        0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,
        7,4,13,1,10,6,15,3,12,0,9,5,2,14,11,8,
        3,10,14,4,9,15,8,1,2,7,0,6,13,11,5,12,
        1,9,11,10,0,8,12,4,13,3,7,15,14,5,6,2,
        4,0,5,9,7,12,2,10,14,1,3,8,11,6,15,13};
    static const u8 RR[80] = {
        5,14,7,0,9,2,11,4,13,6,15,8,1,10,3,12,
        6,11,3,7,0,13,5,10,14,15,8,12,4,9,1,2,
        15,5,1,3,7,14,6,9,11,8,12,2,10,0,4,13,
        8,6,4,1,3,11,15,0,5,12,2,13,9,7,10,14,
        12,15,10,4,1,5,8,7,6,2,13,14,0,3,9,11};
    static const u8 SL[80] = {
        11,14,15,12,5,8,7,9,11,13,14,15,6,7,9,8,
        7,6,8,13,11,9,7,15,7,12,15,9,11,7,13,12,
        11,13,6,7,14,9,13,15,14,8,13,6,5,12,7,5,
        11,12,14,15,14,15,9,8,9,14,5,6,8,6,5,12,
        9,15,5,11,6,8,13,12,5,12,13,14,11,8,5,6};
    static const u8 SR[80] = {
        8,9,9,11,13,15,15,5,7,7,8,11,14,14,12,6,
        9,13,15,7,12,8,9,11,7,7,12,7,6,15,13,11,
        9,7,15,11,8,6,6,14,12,13,5,14,13,13,7,5,
        15,5,8,11,14,14,6,14,6,9,12,9,12,5,15,8,
        8,5,12,9,12,5,14,6,8,13,6,5,15,13,11,11};

    u32 h0=0x67452301, h1=0xefcdab89, h2=0x98badcfe, h3=0x10325476, h4=0xc3d2e1f0;

    // Build a padded message in a local buffer. Our only caller hashes 32 bytes,
    // so a small fixed buffer (one or two 64-byte blocks) is plenty.
    u8 block[128];
    u32 nblocks = ((len + 8) / 64) + 1; // room for 0x80 + 8-byte length
    // nblocks is 1 for len<=55, else 2. Cap defensively.
    if (nblocks > 2) nblocks = 2;
    u32 total = nblocks * 64;
    for (u32 i = 0; i < total; i++) block[i] = (i < len) ? msg[i] : 0;
    block[len] = 0x80;
    u64 bits = (u64)len * 8;
    for (int i = 0; i < 8; i++) block[total - 8 + i] = (u8)(bits >> (i*8)); // little-endian

    for (u32 b = 0; b < nblocks; b++) {
        const u8* blk = block + b*64;
        u32 X[16];
        for (int i = 0; i < 16; i++)
            X[i] = (u32)blk[i*4] | ((u32)blk[i*4+1]<<8) | ((u32)blk[i*4+2]<<16) | ((u32)blk[i*4+3]<<24);

        u32 al=h0,bl=h1,cl=h2,dl=h3,el=h4;
        u32 ar=h0,br=h1,cr=h2,dr=h3,er=h4;
        for (int j = 0; j < 80; j++) {
            u32 t = rotl32(al + rmd_f(j, bl, cl, dl) + X[RL[j]] + KL[j/16], SL[j]) + el;
            al=el; el=dl; dl=rotl32(cl,10); cl=bl; bl=t;
            t = rotl32(ar + rmd_f(79-j, br, cr, dr) + X[RR[j]] + KR[j/16], SR[j]) + er;
            ar=er; er=dr; dr=rotl32(cr,10); cr=br; br=t;
        }
        u32 tmp = h1 + cl + dr;
        h1 = h2 + dl + er;
        h2 = h3 + el + ar;
        h3 = h4 + al + br;
        h4 = h0 + bl + cr;
        h0 = tmp;
    }

    u32 hh[5] = {h0,h1,h2,h3,h4};
    for (int i = 0; i < 5; i++) {
        out[i*4]   = (u8)(hh[i]);
        out[i*4+1] = (u8)(hh[i] >> 8);
        out[i*4+2] = (u8)(hh[i] >> 16);
        out[i*4+3] = (u8)(hh[i] >> 24);
    }
}

// ===========================================================================
// HMAC-SHA512 (RFC 2104) with precomputed inner/outer midstates.
//
// The key-dependent first block of each SHA-512 (ipad/opad) is hashed once in
// hmac512_init; every subsequent MAC only streams the message. PBKDF2 reuses the
// same key 2048x, so this is the key optimization.
// ===========================================================================

typedef struct { sha512_ctx inner; sha512_ctx outer; } hmac512_ctx;

__device__ void hmac512_init(hmac512_ctx* h, const u8* key, u32 keylen) {
    u8 k[128];
    if (keylen > 128) {
        u8 t[64]; sha512(key, keylen, t);
        for (int i = 0; i < 64; i++) k[i] = t[i];
        for (int i = 64; i < 128; i++) k[i] = 0;
    } else {
        for (u32 i = 0; i < keylen; i++) k[i] = key[i];
        for (u32 i = keylen; i < 128; i++) k[i] = 0;
    }
    u8 pad[128];
    for (int i = 0; i < 128; i++) pad[i] = k[i] ^ 0x36;
    sha512_init(&h->inner); sha512_update(&h->inner, pad, 128);
    for (int i = 0; i < 128; i++) pad[i] = k[i] ^ 0x5c;
    sha512_init(&h->outer); sha512_update(&h->outer, pad, 128);
}

// MAC of a single message using the precomputed states (states are copied, not
// mutated, so the same ctx can be reused for many messages).
__device__ void hmac512_compute(const hmac512_ctx* h, const u8* msg, u32 msglen, u8 out[64]) {
    sha512_ctx in = h->inner;
    sha512_update(&in, msg, msglen);
    u8 ih[64]; sha512_final(&in, ih);
    sha512_ctx ou = h->outer;
    sha512_update(&ou, ih, 64);
    sha512_final(&ou, out);
}

// Optimized HMAC for fixed 64-byte message (used in PBKDF2 inner loop).
// Avoids sha512_update branching and copies by computing the padding block directly.
// The inner precomputed state already consumed the 128-byte ipad block.
// Now we append the 64-byte message + SHA-512 padding (1 block = 128 bytes total),
// then feed the 64-byte inner hash into the precomputed outer state.
__device__ __forceinline__ void hmac512_compute_64(
        const hmac512_ctx* h, const u8 msg[64], u8 out[64]) {
    // ── Inner: init from precomputed midstate, then one 128-byte block ──
    // Block = msg[64] || 0x80 || zeros[55] || bitlen(64+128=192 bytes = 1536 bits)
    // 1536 in big-endian u128 = 0x0000000000000600 in the low 64 bits.
    u8 blk[128];
    for (int i = 0;  i < 64; i++) blk[i] = msg[i];
    blk[64] = 0x80;
    for (int i = 65; i < 120; i++) blk[i] = 0;
    // Length = (128 + 64) * 8 = 1536 bits = 0x600, stored as 128-bit big-endian.
    // High 8 bytes = 0, low 8 bytes = 0x0000000000000600.
    blk[120] = 0; blk[121] = 0; blk[122] = 0; blk[123] = 0;
    blk[124] = 0; blk[125] = 0; blk[126] = 0x06; blk[127] = 0x00;

    u64 ih[8];
    for (int i = 0; i < 8; i++) ih[i] = h->inner.h[i];
    sha512_transform(ih, blk);

    // Serialise inner hash to bytes.
    u8 ih_bytes[64];
    for (int i = 0; i < 8; i++)
        for (int j = 0; j < 8; j++) ih_bytes[i*8+j] = (u8)(ih[i] >> (56 - j*8));

    // ── Outer: same idea — precomputed midstate + 64-byte inner hash ──
    for (int i = 0;  i < 64; i++) blk[i] = ih_bytes[i];
    blk[64] = 0x80;
    for (int i = 65; i < 120; i++) blk[i] = 0;
    blk[120] = 0; blk[121] = 0; blk[122] = 0; blk[123] = 0;
    blk[124] = 0; blk[125] = 0; blk[126] = 0x06; blk[127] = 0x00;

    u64 oh[8];
    for (int i = 0; i < 8; i++) oh[i] = h->outer.h[i];
    sha512_transform(oh, blk);

    for (int i = 0; i < 8; i++)
        for (int j = 0; j < 8; j++) out[i*8+j] = (u8)(oh[i] >> (56 - j*8));
}

__device__ void hmac_sha512(const u8* key, u32 keylen, const u8* msg, u32 msglen, u8 out[64]) {
    hmac512_ctx h; hmac512_init(&h, key, keylen);
    hmac512_compute(&h, msg, msglen, out);
}

// ===========================================================================
// PBKDF2-HMAC-SHA512, specialized to dkLen == 64 (one output block), as used by
// BIP-39 seed derivation (salt = "mnemonic" || passphrase, c = 2048).
// ===========================================================================

// IMPORTANT: __noinline__ is required (here and on seed_to_hash160). When both
// are inlined into the single huge k_pipeline frame, nvcc -O miscompiles and
// pbkdf2 produces a wrong seed (verified: the standalone kernels are correct, but
// the inlined combination corrupts the result). Keeping each as its own frame
// matches the individually-verified kernels bit-for-bit. Do not remove.
__device__ __noinline__ void pbkdf2_hmac_sha512_64(
    const u8* pw, u32 pwlen, const u8* salt, u32 saltlen, u32 iters, u8 out[64]) {
    hmac512_ctx h; hmac512_init(&h, pw, pwlen);

    // U1 = HMAC(pw, salt || INT32BE(1))
    u8 u[64];
    {
        sha512_ctx in = h.inner;
        sha512_update(&in, salt, saltlen);
        u8 idx[4] = {0, 0, 0, 1};
        sha512_update(&in, idx, 4);
        u8 ih[64]; sha512_final(&in, ih);
        sha512_ctx ou = h.outer;
        sha512_update(&ou, ih, 64);
        sha512_final(&ou, u);
    }
    for (int i = 0; i < 64; i++) out[i] = u[i];

    // Inner loop: use fixed-64-byte optimized HMAC to skip branching overhead.
    // hmac512_compute_64 assumes msg is always exactly 64 bytes, which is true
    // for all iterations after U1 (each U[i] is the 64-byte output of the previous).
    for (u32 iter = 1; iter < iters; iter++) {
        u8 t[64];
        hmac512_compute_64(&h, u, t);
        for (int i = 0; i < 64; i++) { out[i] ^= t[i]; u[i] = t[i]; }
    }
}

// ===========================================================================
// secp256k1
//
// Field elements are 4x u64 limbs, little-endian: value = n[0] + n[1]*2^64 +
// n[2]*2^128 + n[3]*2^192. p = 2^256 - 0x1000003D1. Points use Jacobian
// coordinates (X,Y,Z) with the point at infinity represented by Z == 0.
// ===========================================================================

typedef struct { u64 n[4]; } fe;   // field element mod p
typedef struct { u64 n[4]; } scalar; // integer mod n (group order)
typedef struct { fe X, Y, Z; } jpoint;

// p (little-endian limbs)
__device__ __constant__ u64 P[4] = {
    0xFFFFFFFEFFFFFC2FULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL};
// n, the group order (little-endian limbs)
__device__ __constant__ u64 N[4] = {
    0xBFD25E8CD0364141ULL, 0xBAAEDCE6AF48A03BULL, 0xFFFFFFFFFFFFFFFEULL, 0xFFFFFFFFFFFFFFFFULL};
// Generator G in affine coordinates.
__device__ __constant__ u64 GX[4] = {
    0x59F2815B16F81798ULL, 0x029BFCDB2DCE28D9ULL, 0x55A06295CE870B07ULL, 0x79BE667EF9DCBBACULL};
__device__ __constant__ u64 GY[4] = {
    0x9C47D08FFB10D4B8ULL, 0xFD17B448A6855419ULL, 0x5DA4FBFC0E1108A8ULL, 0x483ADA7726A3C465ULL};

#define FE_C 0x1000003D1ULL   // p = 2^256 - FE_C

__device__ __forceinline__ void fe_set(fe* r, const u64 v[4]) {
    r->n[0]=v[0]; r->n[1]=v[1]; r->n[2]=v[2]; r->n[3]=v[3];
}
__device__ __forceinline__ void fe_zero(fe* r) { r->n[0]=r->n[1]=r->n[2]=r->n[3]=0; }
__device__ __forceinline__ int fe_is_zero(const fe* a) {
    return (a->n[0]|a->n[1]|a->n[2]|a->n[3]) == 0;
}
__device__ __forceinline__ void fe_one(fe* r) { r->n[0]=1; r->n[1]=r->n[2]=r->n[3]=0; }

// returns 1 if a >= b (treating both as 256-bit little-endian)
__device__ int ge256(const u64 a[4], const u64 b[4]) {
    for (int i = 3; i >= 0; i--) {
        if (a[i] != b[i]) return a[i] > b[i];
    }
    return 1; // equal
}

// r = a - m (assumes a >= m), 256-bit
__device__ void sub256(u64 r[4], const u64 a[4], const u64 m[4]) {
    unsigned __int128 borrow = 0;
    for (int i = 0; i < 4; i++) {
        unsigned __int128 cur = (unsigned __int128)a[i] - m[i] - borrow;
        r[i] = (u64)cur;
        borrow = (cur >> 64) & 1; // 1 if underflow
    }
}

__device__ __forceinline__ void fe_reduce_p(fe* r) {
    if (ge256(r->n, P)) sub256(r->n, r->n, P);
}

__device__ void fe_add(fe* r, const fe* a, const fe* b) {
    unsigned __int128 carry = 0;
    for (int i = 0; i < 4; i++) {
        unsigned __int128 cur = (unsigned __int128)a->n[i] + b->n[i] + carry;
        r->n[i] = (u64)cur; carry = cur >> 64;
    }
    // value = r + carry*2^256 ≡ r + carry*FE_C (mod p)
    if (carry) {
        unsigned __int128 c2 = 0;
        unsigned __int128 cur = (unsigned __int128)r->n[0] + (unsigned __int128)carry*FE_C;
        r->n[0]=(u64)cur; c2=cur>>64;
        for (int i = 1; i < 4; i++) { cur=(unsigned __int128)r->n[i]+c2; r->n[i]=(u64)cur; c2=cur>>64; }
        if (c2) { // extremely rare second wrap
            cur=(unsigned __int128)r->n[0]+(unsigned __int128)c2*FE_C; r->n[0]=(u64)cur; c2=cur>>64;
            for (int i=1;i<4;i++){cur=(unsigned __int128)r->n[i]+c2;r->n[i]=(u64)cur;c2=cur>>64;}
        }
    }
    fe_reduce_p(r);
}

__device__ void fe_sub(fe* r, const fe* a, const fe* b) {
    // r = a - b mod p; if underflow add p
    unsigned __int128 borrow = 0;
    u64 t[4];
    for (int i = 0; i < 4; i++) {
        unsigned __int128 cur = (unsigned __int128)a->n[i] - b->n[i] - borrow;
        t[i] = (u64)cur; borrow = (cur >> 64) & 1;
    }
    if (borrow) {
        unsigned __int128 carry = 0;
        for (int i = 0; i < 4; i++) {
            unsigned __int128 cur = (unsigned __int128)t[i] + P[i] + carry;
            t[i] = (u64)cur; carry = cur >> 64;
        }
    }
    r->n[0]=t[0]; r->n[1]=t[1]; r->n[2]=t[2]; r->n[3]=t[3];
}

// reduce a 512-bit product (8 little-endian limbs) mod p into r
__device__ void fe_reduce512(fe* r, const u64 t[8]) {
    // m[0..3] = t_lo + t_hi*FE_C, m[4] = carry
    u64 m[5];
    unsigned __int128 carry = 0;
    for (int i = 0; i < 4; i++) {
        unsigned __int128 cur = (unsigned __int128)t[4+i]*FE_C + t[i] + carry;
        m[i] = (u64)cur; carry = cur >> 64;
    }
    m[4] = (u64)carry;
    // fold m[4]*FE_C back in
    unsigned __int128 cur = (unsigned __int128)m[4]*FE_C + m[0];
    r->n[0] = (u64)cur; carry = cur >> 64;
    for (int i = 1; i < 4; i++) { cur = (unsigned __int128)m[i] + carry; r->n[i] = (u64)cur; carry = cur >> 64; }
    u64 extra = (u64)carry; // 0 or 1
    if (extra) {
        cur = (unsigned __int128)r->n[0] + (unsigned __int128)extra*FE_C; r->n[0]=(u64)cur; carry=cur>>64;
        for (int i = 1; i < 4; i++) { cur=(unsigned __int128)r->n[i]+carry; r->n[i]=(u64)cur; carry=cur>>64; }
    }
    fe_reduce_p(r);
}

__device__ void fe_mul(fe* r, const fe* a, const fe* b) {
    u64 t[8]; for (int i = 0; i < 8; i++) t[i] = 0;
    for (int i = 0; i < 4; i++) {
        unsigned __int128 carry = 0;
        for (int j = 0; j < 4; j++) {
            unsigned __int128 cur = (unsigned __int128)a->n[i]*b->n[j] + t[i+j] + carry;
            t[i+j] = (u64)cur; carry = cur >> 64;
        }
        t[i+4] = (u64)carry;
    }
    fe_reduce512(r, t);
}

__device__ __forceinline__ void fe_sqr(fe* r, const fe* a) { fe_mul(r, a, a); }

// modular inverse via Fermat: a^(p-2) mod p. p-2 little-endian limbs:
__device__ void fe_inv(fe* r, const fe* a) {
    // exponent = p - 2 = 0xFFFF...FFFFFFFEFFFFFC2D (lo limb), rest 0xFFFF...
    const u64 e[4] = {0xFFFFFFFEFFFFFC2DULL, 0xFFFFFFFFFFFFFFFFULL,
                      0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL};
    fe result; fe_one(&result);
    fe base = *a;
    for (int limb = 0; limb < 4; limb++) {
        u64 w = e[limb];
        int bits = (limb == 3) ? 64 : 64; // process all 64 bits each limb
        for (int b = 0; b < bits; b++) {
            if ((w >> b) & 1) fe_mul(&result, &result, &base);
            fe_sqr(&base, &base);
        }
    }
    *r = result;
}

// ---- point operations (Jacobian) ----

__device__ __forceinline__ void jp_set_infinity(jpoint* p) { fe_zero(&p->Z); }
__device__ __forceinline__ int jp_is_infinity(const jpoint* p) { return fe_is_zero(&p->Z); }

// P3 = 2*P1 (dbl-2009-l, a=0). Writes to locals first so r may alias p.
__device__ void jp_double(jpoint* r, const jpoint* p) {
    if (jp_is_infinity(p) || fe_is_zero(&p->Y)) { jp_set_infinity(r); return; }
    fe A, B, C, D, E, F, t, t2, X3, Y3, Z3;
    fe_sqr(&A, &p->X);                 // A = X^2
    fe_sqr(&B, &p->Y);                 // B = Y^2
    fe_sqr(&C, &B);                    // C = B^2
    fe_add(&t, &p->X, &B); fe_sqr(&t, &t);   // (X+B)^2
    fe_sub(&t, &t, &A); fe_sub(&t, &t, &C);  // (X+B)^2 - A - C
    fe_add(&D, &t, &t);                // D = 2*((X+B)^2 - A - C)
    fe_add(&E, &A, &A); fe_add(&E, &E, &A);  // E = 3*A
    fe_sqr(&F, &E);                    // F = E^2
    fe_mul(&t, &p->Y, &p->Z); fe_add(&Z3, &t, &t); // Z3 = 2*Y*Z  (before Y is touched)
    fe_add(&t, &D, &D);                // 2D
    fe_sub(&X3, &F, &t);               // X3 = F - 2D
    fe_sub(&t, &D, &X3);               // D - X3
    fe_mul(&t, &E, &t);                // E*(D - X3)
    fe_add(&t2, &C, &C); fe_add(&t2, &t2, &t2); fe_add(&t2, &t2, &t2); // 8C
    fe_sub(&Y3, &t, &t2);              // Y3 = E*(D-X3) - 8C
    r->X = X3; r->Y = Y3; r->Z = Z3;
}

// P3 = P1 + Q where Q is affine (qx,qy). (madd-2007-bl). Handles P1 = infinity
// and the P1 == +/-Q cases.
__device__ void jp_add_affine(jpoint* r, const jpoint* p, const fe* qx, const fe* qy) {
    if (jp_is_infinity(p)) { r->X=*qx; r->Y=*qy; fe_one(&r->Z); return; }
    fe Z1Z1, U2, S2, H, HH, I, J, rr, V, t, t2;
    fe_sqr(&Z1Z1, &p->Z);              // Z1Z1 = Z1^2
    fe_mul(&U2, qx, &Z1Z1);            // U2 = X2*Z1Z1
    fe_mul(&S2, qy, &p->Z); fe_mul(&S2, &S2, &Z1Z1); // S2 = Y2*Z1^3
    fe_sub(&H, &U2, &p->X);            // H = U2 - X1
    fe_sub(&t, &S2, &p->Y);            // S2 - Y1
    if (fe_is_zero(&H)) {
        if (fe_is_zero(&t)) { jp_double(r, p); return; } // P == Q
        jp_set_infinity(r); return;                       // P == -Q
    }
    fe_sqr(&HH, &H);                   // HH = H^2
    fe_add(&I, &HH, &HH); fe_add(&I, &I, &I); // I = 4*HH
    fe_mul(&J, &H, &I);                // J = H*I
    fe_add(&rr, &t, &t);               // r = 2*(S2 - Y1)
    fe_mul(&V, &p->X, &I);             // V = X1*I
    fe X3, Y3, Z3;
    fe_sqr(&X3, &rr);                  // r^2
    fe_sub(&X3, &X3, &J);              // r^2 - J
    fe_add(&t2, &V, &V); fe_sub(&X3, &X3, &t2); // X3 = r^2 - J - 2V
    fe_sub(&t, &V, &X3);               // V - X3
    fe_mul(&t, &rr, &t);               // r*(V - X3)
    fe_mul(&t2, &p->Y, &J); fe_add(&t2, &t2, &t2); // 2*Y1*J
    fe_sub(&Y3, &t, &t2);              // Y3 = r*(V-X3) - 2*Y1*J
    // Z3 = (Z1+H)^2 - Z1Z1 - HH = 2*Z1*H
    fe_add(&t, &p->Z, &H); fe_sqr(&t, &t);
    fe_sub(&t, &t, &Z1Z1); fe_sub(&Z3, &t, &HH);
    r->X = X3; r->Y = Y3; r->Z = Z3;
}

// Convert big-endian 32-byte scalar to little-endian limbs.
__device__ void be32_to_limbs(const u8* b, u64 out[4]) {
    for (int i = 0; i < 4; i++) {
        u64 v = 0;
        for (int j = 0; j < 8; j++) v = (v << 8) | b[i*8 + j];
        out[3 - i] = v;
    }
}

// R = k*G, k given as little-endian limbs (k must be in [1, n-1]).
__device__ void scalar_mul_G(jpoint* r, const u64 k[4]) {
    jpoint acc; jp_set_infinity(&acc);
    fe gx, gy; fe_set(&gx, GX); fe_set(&gy, GY);
    for (int limb = 3; limb >= 0; limb--) {
        for (int b = 63; b >= 0; b--) {
            jp_double(&acc, &acc);
            if ((k[limb] >> b) & 1) jp_add_affine(&acc, &acc, &gx, &gy);
        }
    }
    *r = acc;
}

// Serialize k*G as a 33-byte compressed pubkey.
__device__ void pubkey_compressed(const u64 k[4], u8 out[33]) {
    jpoint p; scalar_mul_G(&p, k);
    if (jp_is_infinity(&p)) { for (int i=0;i<33;i++) out[i]=0; return; }
    fe zinv, zinv2, zinv3, x, y;
    fe_inv(&zinv, &p.Z);
    fe_sqr(&zinv2, &zinv);
    fe_mul(&zinv3, &zinv2, &zinv);
    fe_mul(&x, &p.X, &zinv2);
    fe_mul(&y, &p.Y, &zinv3);
    out[0] = 0x02 | (u8)(y.n[0] & 1);
    // x big-endian
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 8; j++) out[1 + i*8 + j] = (u8)(x.n[3-i] >> (56 - j*8));
}

// r = (a + b) mod n, all as big-endian 32-byte; returns 1 if result != 0.
__device__ void scalar_add_modn(const u64 a[4], const u64 b[4], u64 r[4]) {
    unsigned __int128 carry = 0;
    for (int i = 0; i < 4; i++) {
        unsigned __int128 cur = (unsigned __int128)a[i] + b[i] + carry;
        r[i] = (u64)cur; carry = cur >> 64;
    }
    if (carry || ge256(r, N)) sub256(r, r, N);
}

// ===========================================================================
// BIP32 derivation of the fixed path m/44'/0'/0'/0/0, ending in the P2PKH
// hash160 of the resulting public key.
// ===========================================================================

__device__ void limbs_to_be32(const u64 k[4], u8* out) {
    for (int j = 0; j < 4; j++)
        for (int b = 0; b < 8; b++) out[j*8 + b] = (u8)(k[3-j] >> (56 - b*8));
}

// CKD-priv for one level. key/cc are 32-byte big-endian buffers, updated in place.
__device__ void bip32_ckd_priv(u8 key[32], u8 cc[32], u32 index) {
    u8 data[37];
    if (index & 0x80000000u) {
        // hardened: 0x00 || ser256(key) || ser32(index)
        data[0] = 0x00;
        for (int i = 0; i < 32; i++) data[1+i] = key[i];
    } else {
        // normal: serP(point(key)) || ser32(index)
        u64 k[4]; be32_to_limbs(key, k);
        u8 pub[33]; pubkey_compressed(k, pub);
        for (int i = 0; i < 33; i++) data[i] = pub[i];
    }
    data[33] = (u8)(index >> 24); data[34] = (u8)(index >> 16);
    data[35] = (u8)(index >> 8);  data[36] = (u8)(index);

    u8 I[64];
    hmac_sha512(cc, 32, data, 37, I);

    // child key = (IL + parent key) mod n
    u64 il[4], pk[4], ck[4];
    be32_to_limbs(I, il);
    be32_to_limbs(key, pk);
    scalar_add_modn(il, pk, ck);
    limbs_to_be32(ck, key);
    // child chain code = IR
    for (int i = 0; i < 32; i++) cc[i] = I[32 + i];
}

// seed[64] -> raw 32-byte private key at m/44'/0'/0'/0/0.
// Stops before pubkey derivation. Used by k_pipeline_pubkey to skip
// the SHA256+RIPEMD160 step and compare the pubkey directly instead.
// __noinline__ required — see the note on pbkdf2_hmac_sha512_64.
__device__ __noinline__ void seed_to_privkey(const u8 seed[64], u8 privkey[32]) {
    u8 I[64];
    hmac_sha512((const u8*)"Bitcoin seed", 12, seed, 64, I);
    u8 key[32], cc[32];
    for (int i = 0; i < 32; i++) { key[i] = I[i]; cc[i] = I[32 + i]; }

    const u32 H = 0x80000000u;
    bip32_ckd_priv(key, cc, H + 44);
    bip32_ckd_priv(key, cc, H +  0);
    bip32_ckd_priv(key, cc, H +  0);
    bip32_ckd_priv(key, cc,      0);
    bip32_ckd_priv(key, cc,      0);

    for (int i = 0; i < 32; i++) privkey[i] = key[i];
}

// seed[64] -> P2PKH hash160[20] for m/44'/0'/0'/0/0.
// __noinline__ required — see the note on pbkdf2_hmac_sha512_64.
__device__ __noinline__ void seed_to_hash160(const u8 seed[64], u8 hash160[20]) {
    u8 I[64];
    hmac_sha512((const u8*)"Bitcoin seed", 12, seed, 64, I);
    u8 key[32], cc[32];
    for (int i = 0; i < 32; i++) { key[i] = I[i]; cc[i] = I[32 + i]; }

    const u32 H = 0x80000000u;
    bip32_ckd_priv(key, cc, H + 44);
    bip32_ckd_priv(key, cc, H + 0);
    bip32_ckd_priv(key, cc, H + 0);
    bip32_ckd_priv(key, cc, 0);
    bip32_ckd_priv(key, cc, 0);

    u64 k[4]; be32_to_limbs(key, k);
    u8 pub[33]; pubkey_compressed(k, pub);
    u8 sha[32]; sha256(pub, 33, sha);
    ripemd160(sha, 32, hash160);
}

// ===========================================================================
// Full search kernel: one candidate (12 word indices) per thread.
// Filters on the BIP-39 checksum, derives the seed and address, compares to the
// target hash160, and records the first match via atomicCAS.
// ===========================================================================

// Largest joined mnemonic ("word word ... word", 12 words + 11 spaces). 512 is
// comfortably above every BIP-39 language (host asserts the real bound fits).
#define MNEMONIC_BUF 512

// Returns 1 if the 12 indices form a valid BIP-39 checksum.
__device__ int bip39_checksum_ok(const unsigned short idx[12]) {
    u8 ent[17];
    for (int i = 0; i < 17; i++) ent[i] = 0;
    int bitpos = 0;
    for (int w = 0; w < 12; w++) {
        u32 v = idx[w]; // 11-bit word index
        for (int b = 10; b >= 0; b--) {
            if ((v >> b) & 1) ent[bitpos >> 3] |= (u8)(0x80 >> (bitpos & 7));
            bitpos++;
        }
    }
    // entropy = ent[0..16]; 4-bit checksum = high nibble of ent[16]
    u8 h[32]; sha256(ent, 16, h);
    return (ent[16] >> 4) == (h[0] >> 4);
}

// Pass 1: keep only candidates with a valid BIP-39 checksum (~1/16). Survivor
// candidate indices are compacted into `survivors` via an atomic counter, so the
// heavy second pass runs with no warp divergence.
extern "C" __global__
void k_filter(const unsigned short* cand, u32 n, u32* survivors, u32* counter) {
    u32 i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (bip39_checksum_ok(cand + (u64)i * 12)) {
        u32 slot = atomicAdd(counter, 1u);
        survivors[slot] = i;
    }
}

// Pass 2: full derivation for each compacted survivor. Thread t handles
// survivors[t]; every thread does real work.
extern "C" __global__
void k_pipeline(const unsigned short* cand, const u32* survivors, u32 count,
                const u8* wordlist, const u8* word_lens, u32 word_stride,
                const u8* target_h160,
                unsigned int* found_flag, unsigned int* found_idx) {
    u32 t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= count) return;
    if (*found_flag) return;

    u32 i = survivors[t];
    const unsigned short* idx = cand + (u64)i * 12;

    // Build the (already NFKD) mnemonic: words joined by single spaces.
    u8 msg[MNEMONIC_BUF];
    u32 mlen = 0;
    for (int w = 0; w < 12; w++) {
        u32 wi = idx[w];
        u8 wl = word_lens[wi];
        const u8* wp = wordlist + (u64)wi * word_stride;
        for (u32 c = 0; c < wl; c++) msg[mlen++] = wp[c];
        if (w < 11) msg[mlen++] = ' ';
    }

    u8 seed[64];
    const u8 salt[8] = {'m','n','e','m','o','n','i','c'};
    pbkdf2_hmac_sha512_64(msg, mlen, salt, 8, 2048, seed);

    u8 h160[20];
    seed_to_hash160(seed, h160);

    int eq = 1;
    for (int j = 0; j < 20; j++) if (h160[j] != target_h160[j]) { eq = 0; break; }
    if (eq) {
        if (atomicCAS(found_flag, 0u, 1u) == 0u) *found_idx = i;
    }
}

// ===========================================================================
// Pubkey pipeline: same flow as k_pipeline but stops at the compressed public
// key and compares 33 bytes against target_pubkey instead of computing
// SHA256(pubkey) + RIPEMD160. Saves ~2 hash rounds per surviving candidate.
//
// Pass 1 (k_filter) is shared — BIP-39 checksum is always checked first.
// ===========================================================================
extern "C" __global__
void k_pipeline_pubkey(
        const unsigned short* cand,    // flat [N×12] word-index array
        const u32*            survivors, // compacted survivor indices from k_filter
        u32                   count,     // number of survivors
        const u8*             wordlist,
        const u8*             word_lens,
        u32                   word_stride,
        const u8*             target_pubkey, // 33 bytes: compressed pubkey to match
        unsigned int*         found_flag,
        unsigned int*         found_idx)
{
    u32 t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= count) return;
    if (*found_flag) return;

    u32 i = survivors[t];
    const unsigned short* idx = cand + (u64)i * 12;

    // Build the NFKD mnemonic string (same as k_pipeline).
    u8 msg[MNEMONIC_BUF];
    u32 mlen = 0;
    for (int w = 0; w < 12; w++) {
        u32 wi = idx[w];
        u8 wl  = word_lens[wi];
        const u8* wp = wordlist + (u64)wi * word_stride;
        for (u32 c = 0; c < wl; c++) msg[mlen++] = wp[c];
        if (w < 11) msg[mlen++] = ' ';
    }

    // BIP-39 seed via PBKDF2-HMAC-SHA512 (2048 iterations, no passphrase).
    u8 seed[64];
    const u8 salt[8] = {'m','n','e','m','o','n','i','c'};
    pbkdf2_hmac_sha512_64(msg, mlen, salt, 8, 2048, seed);

    // BIP32 m/44'/0'/0'/0/0 -> private key (stops before hash160).
    u8 privkey[32];
    seed_to_privkey(seed, privkey);

    // Derive the compressed 33-byte public key.
    u64 k[4]; be32_to_limbs(privkey, k);
    u8 pub[33]; pubkey_compressed(k, pub);

    // Compare 33-byte compressed pubkey against target.
    int eq = 1;
    for (int j = 0; j < 33; j++) if (pub[j] != target_pubkey[j]) { eq = 0; break; }
    if (eq) {
        if (atomicCAS(found_flag, 0u, 1u) == 0u) *found_idx = i;
    }
}

// ===========================================================================
// Selftest kernels: one input message per thread (stride-packed), one digest out.
// ===========================================================================

extern "C" __global__
void k_sha256(const u8* msgs, const u32* lens, u32 stride, u8* out, u32 n) {
    u32 i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    sha256(msgs + (u64)i*stride, lens[i], out + (u64)i*32);
}

extern "C" __global__
void k_sha512(const u8* msgs, const u32* lens, u32 stride, u8* out, u32 n) {
    u32 i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    sha512(msgs + (u64)i*stride, lens[i], out + (u64)i*64);
}

extern "C" __global__
void k_ripemd160(const u8* msgs, const u32* lens, u32 stride, u8* out, u32 n) {
    u32 i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    ripemd160(msgs + (u64)i*stride, lens[i], out + (u64)i*20);
}

extern "C" __global__
void k_hmac_sha512(const u8* keys, const u32* klens, u32 kstride,
                   const u8* msgs, const u32* mlens, u32 mstride,
                   u8* out, u32 n) {
    u32 i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    hmac_sha512(keys + (u64)i*kstride, klens[i], msgs + (u64)i*mstride, mlens[i], out + (u64)i*64);
}

extern "C" __global__
void k_pbkdf2(const u8* pws, const u32* pwlens, u32 pwstride,
              const u8* salts, const u32* saltlens, u32 sstride,
              u32 iters, u8* out, u32 n) {
    u32 i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    pbkdf2_hmac_sha512_64(pws + (u64)i*pwstride, pwlens[i],
                          salts + (u64)i*sstride, saltlens[i], iters, out + (u64)i*64);
}

// One 32-byte big-endian private key in -> 33-byte compressed pubkey out.
extern "C" __global__
void k_pubkey(const u8* privs, u8* out, u32 n) {
    u32 i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    u64 k[4]; be32_to_limbs(privs + (u64)i*32, k);
    pubkey_compressed(k, out + (u64)i*33);
}

// (a * b) mod p, all 32-byte big-endian. Debug helper for field arithmetic.
extern "C" __global__
void k_fe_mul(const u8* a, const u8* b, u8* out, u32 n) {
    u32 i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    u64 la[4], lb[4];
    be32_to_limbs(a + (u64)i*32, la);
    be32_to_limbs(b + (u64)i*32, lb);
    fe fa, fb, fr; fe_set(&fa, la); fe_set(&fb, lb);
    fe_mul(&fr, &fa, &fb);
    u8* o = out + (u64)i*32;
    for (int j = 0; j < 4; j++)
        for (int k = 0; k < 8; k++) o[j*8 + k] = (u8)(fr.n[3-j] >> (56 - k*8));
}

// One 64-byte seed in -> 20-byte P2PKH hash160 (m/44'/0'/0'/0/0) out.
extern "C" __global__
void k_seed_to_hash160(const u8* seeds, u8* out, u32 n) {
    u32 i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    seed_to_hash160(seeds + (u64)i*64, out + (u64)i*20);
}

// a^{-1} mod p, 32-byte big-endian. Debug helper.
extern "C" __global__
void k_fe_inv(const u8* a, u8* out, u32 n) {
    u32 i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    u64 la[4]; be32_to_limbs(a + (u64)i*32, la);
    fe fa, fr; fe_set(&fa, la); fe_inv(&fr, &fa);
    u8* o = out + (u64)i*32;
    for (int j = 0; j < 4; j++)
        for (int k = 0; k < 8; k++) o[j*8 + k] = (u8)(fr.n[3-j] >> (56 - k*8));
}

// (a + b) mod n, all 32-byte big-endian.
extern "C" __global__
void k_scalar_add(const u8* a, const u8* b, u8* out, u32 n) {
    u32 i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    u64 la[4], lb[4], lr[4];
    be32_to_limbs(a + (u64)i*32, la);
    be32_to_limbs(b + (u64)i*32, lb);
    scalar_add_modn(la, lb, lr);
    u8* o = out + (u64)i*32;
    for (int j = 0; j < 4; j++)
        for (int k = 0; k < 8; k++) o[j*8 + k] = (u8)(lr[3-j] >> (56 - k*8));
}
