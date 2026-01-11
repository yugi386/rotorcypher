use argon2::{Algorithm, Argon2, Params, Version};
use hmac::{Hmac, Mac};
use rand::RngCore;
use sha3::{
    Sha3_512, Shake256,
    digest::{ExtendableOutput, Update},
};
use std::fmt;
use std::fs::{File, OpenOptions};
use std::io::{self, BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::time::Instant;
use zeroize::Zeroize;

const OUT_SIZE: usize = 1040;
const R5_SIZE: usize = 32;
const KS_BLOCK: usize = 1024;

// HMAC-SHA3-512
type HmacSha3_512 = Hmac<Sha3_512>;

/* =========================================================
 * Erros
 * ========================================================= */

#[derive(Debug)]
pub enum RotorError {
    Argon2Params,
    Argon2Hash,
    PrimeListTooSmall,
    InvalidRotorSizes,
}

impl fmt::Display for RotorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RotorError::Argon2Params => write!(f, "Parâmetros inválidos do Argon2"),
            RotorError::Argon2Hash => write!(f, "Falha ao derivar chave com Argon2"),
            RotorError::PrimeListTooSmall => write!(f, "Lista de primos insuficiente"),
            RotorError::InvalidRotorSizes => write!(f, "Tamanhos de rotores inválidos"),
        }
    }
}
impl std::error::Error for RotorError {}

#[derive(Debug)]
pub enum FileCryptoError {
    Rotor(RotorError),
    Io(io::Error),
    IntegrityMismatch,
    InvalidCiphertextFormat,
}

impl fmt::Display for FileCryptoError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FileCryptoError::Rotor(e) => write!(f, "Erro na cifra de rotores: {}", e),
            FileCryptoError::Io(e) => write!(f, "Erro de I/O: {}", e),
            FileCryptoError::IntegrityMismatch => write!(f, "Falha de integridade (HMAC inválido)"),
            FileCryptoError::InvalidCiphertextFormat => write!(f, "Formato de ciphertext inválido"),
        }
    }
}

impl std::error::Error for FileCryptoError {}

impl From<RotorError> for FileCryptoError {
    fn from(e: RotorError) -> Self {
        FileCryptoError::Rotor(e)
    }
}
impl From<io::Error> for FileCryptoError {
    fn from(e: io::Error) -> Self {
        FileCryptoError::Io(e)
    }
}

/* =========================================================
 * Utilidades
 * ========================================================= */

/// Comparação em tempo constante (evita timing leak de MAC).
fn ct_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff: u8 = 0;
    for i in 0..a.len() {
        diff |= a[i] ^ b[i];
    }
    diff == 0
}

#[inline]
fn mul_mod_65537(a: u16, b: u16) -> u16 {
    // IDEA-style: 0 representa 65536
    let aa: u64 = if a == 0 { 65536 } else { a as u64 };
    let bb: u64 = if b == 0 { 65536 } else { b as u64 };
    let mut m = (aa * bb) % 65537;
    if m == 65536 {
        m = 0;
    }
    m as u16
}

/// Disambiguation helper para leitura do XOF (evita conflito com std::io::Read).
#[inline]
fn xof_read_exact<R: sha3::digest::XofReader>(reader: &mut R, out: &mut [u8]) {
    sha3::digest::XofReader::read(reader, out);
}

/* =========================================================
 * Primos + difusão
 * ========================================================= */

/// Gera lista de primos em [min..=max] (u16), retornando os primeiros `limit`.
fn primes_in_range_u16(min: u16, max: u16, limit: usize) -> Vec<u16> {
    let max_usize = max as usize;
    let mut is_prime = vec![true; max_usize + 1];

    is_prime[0] = false;

    if max_usize >= 1 {
        is_prime[1] = false;
    }

    let mut p = 2usize;
    while p * p <= max_usize {
        if is_prime[p] {
            let mut m = p * p;
            while m <= max_usize {
                is_prime[m] = false;
                m += p;
            }
        }
        p += 1;
    }

    let mut primes = Vec::new();
    for n in (min as usize)..=max_usize {
        if is_prime[n] {
            primes.push(n as u16);
            if primes.len() == limit {
                break;
            }
        }
    }
    primes
}

/// Leitor MSB-first de bits.
struct BitReader<'a> {
    bytes: &'a [u8],
    bitpos: usize, // 0..bytes.len()*8
}
impl<'a> BitReader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, bitpos: 0 }
    }
    fn read_bits(&mut self, n: usize) -> u16 {
        let mut v: u16 = 0;
        for _ in 0..n {
            let byte_index = self.bitpos / 8;
            let bit_index = 7 - (self.bitpos % 8);
            let bit = (self.bytes[byte_index] >> bit_index) & 1;
            v = (v << 1) | (bit as u16);
            self.bitpos += 1;
        }
        v
    }
}

/// Permuta a lista de primos usando 640 chaves de 11 bits (extraídas de 880 bytes = 7040 bits).
fn rotor_order_from_diffusion(diff880: &[u8], primes: &[u16]) -> Vec<u16> {
    debug_assert_eq!(diff880.len(), 880);
    debug_assert_eq!(primes.len(), 640);

    let mut br = BitReader::new(diff880);
    let mut pairs: Vec<(u16, u16)> = primes
        .iter()
        .map(|&p| {
            let key = br.read_bits(11); // 0..2047
            (p, key)
        })
        .collect();

    // Desempate determinístico
    pairs.sort_unstable_by(|(pa, ka), (pb, kb)| ka.cmp(kb).then(pa.cmp(pb)));
    pairs.into_iter().map(|(p, _)| p).collect()
}

/// Extrai 5 tamanhos (17..32) a partir de 3 bytes (6 nibbles).
fn rotor_sizes_from_diffusion(diff3: &[u8]) -> Result<[usize; 5], RotorError> {
    if diff3.len() != 3 {
        return Err(RotorError::InvalidRotorSizes);
    }
    let n0 = (diff3[0] >> 4) & 0x0F;
    let n1 = diff3[0] & 0x0F;
    let n2 = (diff3[1] >> 4) & 0x0F;
    let n3 = diff3[1] & 0x0F;
    let n4 = (diff3[2] >> 4) & 0x0F;

    Ok([
        17 + n0 as usize,
        17 + n1 as usize,
        17 + n2 as usize,
        17 + n3 as usize,
        17 + n4 as usize,
    ])
}

/* =========================================================
 * Estruturas de Rotor
 * ========================================================= */

#[derive(Clone, Debug)]
struct RotorVecMeta {
    offset: usize,
    len: usize,
    cursor: usize,
}

#[derive(Clone, Debug)]
struct Rotor {
    data: Vec<u8>,
    vecs: Vec<RotorVecMeta>,
}

impl Rotor {
    /// Gera `out.len()` bytes: XOR de todos os vetores, avançando todos os cursores.
    fn generate(&mut self, out: &mut [u8]) {
        out.fill(0);

        for v in &mut self.vecs {
            let mut produced = 0usize;

            while produced < out.len() {
                let remaining = out.len() - produced;
                let cursor = v.cursor;
                let avail = v.len - cursor;
                let take = avail.min(remaining);

                let src = &self.data[v.offset + cursor..v.offset + cursor + take];
                xor_inplace(&mut out[produced..produced + take], src);

                produced += take;
                v.cursor = (v.cursor + take) % v.len;
            }
        }
    }
}

/* =========================================================
 * KDF e construção dos rotores
 * ========================================================= */

/// Gera seed 512 bits via Argon2id (parâmetros fixos e explícitos).
fn kdf_argon2id_512(password: &[u8], salt: &[u8]) -> Result<[u8; 64], RotorError> {
    let params = Params::new(
        65_536, // 64 MiB em KiB
        3,      // t_cost
        1,      // p_cost
        Some(64),
    )
    .map_err(|_| RotorError::Argon2Params)?;

    let argon2 = Argon2::new(Algorithm::Argon2id, Version::V0x13, params);

    let mut out = [0u8; 64];
    argon2
        .hash_password_into(password, salt, &mut out)
        .map_err(|_| RotorError::Argon2Hash)?;
    Ok(out)
}

/// Constrói os 5 rotores a partir de seed+nonce usando SHAKE256 (determinístico).
fn build_rotors_from_seed(seed512: &[u8; 64], nonce: &[u8]) -> Result<[Rotor; 5], RotorError> {
    let primes = primes_in_range_u16(300, 10_000, 640);
    if primes.len() < 640 {
        return Err(RotorError::PrimeListTooSmall);
    }

    let mut shake = Shake256::default();
    shake.update(b"ROTORCIPHER-V2");
    shake.update(seed512);
    shake.update(nonce);

    let mut reader = shake.finalize_xof();

    // 880 bytes → ordem; 3 bytes → tamanhos
    let mut cfg = [0u8; 883];
    xof_read_exact(&mut reader, &mut cfg);

    let rotor_order = rotor_order_from_diffusion(&cfg[0..880], &primes);
    let rotor_sizes = rotor_sizes_from_diffusion(&cfg[880..883])?;

    let mut order_it = rotor_order.into_iter();

    let mut rotors: Vec<Rotor> = Vec::with_capacity(5);
    for r in 0..5 {
        let nvec = rotor_sizes[r];
        let mut metas = Vec::with_capacity(nvec);

        // Comprimentos e tamanho total
        let mut lens: Vec<usize> = Vec::with_capacity(nvec);
        let mut total_len = 0usize;
        for _ in 0..nvec {
            let p = order_it.next().ok_or(RotorError::PrimeListTooSmall)? as usize;
            lens.push(p);
            total_len += p;
        }

        let mut data = vec![0u8; total_len];
        let mut offset = 0usize;

        for &len in &lens {
            // Preenche bytes do vetor (XOF)
            xof_read_exact(&mut reader, &mut data[offset..offset + len]);

            // Cursor inicial pseudo-aleatório (2 bytes, XOF)
            let mut cbytes = [0u8; 2];
            xof_read_exact(&mut reader, &mut cbytes);
            let c = u16::from_be_bytes(cbytes) as usize;
            let cursor = c % len;

            metas.push(RotorVecMeta {
                offset,
                len,
                cursor,
            });

            offset += len;
        }

        rotors.push(Rotor { data, vecs: metas });
    }

    // Vec -> array sem unwrap (evita bound Debug)
    let arr: [Rotor; 5] = rotors
        .try_into()
        .map_err(|_| RotorError::PrimeListTooSmall)?;

    Ok(arr)
}

/* =========================================================
 * RotorCipher + Keystream streaming
 * ========================================================= */

pub struct RotorCipher {
    rotors: [Rotor; 5],
    block_counter: u64,
}

impl RotorCipher {
    pub fn new(password: &[u8], salt: &[u8], nonce: &[u8]) -> Result<Self, RotorError> {
        let mut seed = kdf_argon2id_512(password, salt)?;
        let rotors = build_rotors_from_seed(&seed, nonce)?;
        seed.zeroize();
        Ok(Self {
            rotors,
            block_counter: 0,
        })
    }

    /// Gera 1024 bytes de keystream usando SHAKE256 (XOF).
    pub fn keystream_block_1024(&mut self) -> [u8; KS_BLOCK] {
        let mut r1 = [0u8; OUT_SIZE];
        let mut r2 = [0u8; OUT_SIZE];
        let mut r3 = [0u8; OUT_SIZE];
        let mut r4 = [0u8; OUT_SIZE];
        let mut r5 = [0u8; R5_SIZE];

        self.rotors[0].generate(&mut r1);
        self.rotors[1].generate(&mut r2);
        self.rotors[2].generate(&mut r3);
        self.rotors[3].generate(&mut r4);
        self.rotors[4].generate(&mut r5);

        let mut out = [0u8; OUT_SIZE];
        for i in 0..OUT_SIZE {
            let m1 = mul_mod_65537(r1[i] as u16, r2[i] as u16) as u32;
            let m2 = mul_mod_65537(r3[i] as u16, r4[i] as u16) as u32;
            out[i] = ((m1 + m2) & 0xFF) as u8;
        }

        // Rotor 5 → saltos → T[64]
        let mut steps = [0u8; 64];
        for (i, b) in r5.iter().enumerate() {
            steps[2 * i] = (b >> 4) & 0x0F;
            steps[2 * i + 1] = b & 0x0F;
        }

        let mut t = [0u8; 64];
        let mut pos = (steps[0] as usize) % OUT_SIZE;
        t[0] = out[pos];
        for k in 1..64 {
            pos = (pos + steps[k] as usize) % OUT_SIZE;
            t[k] = out[pos];
        }

        // SHAKE256(T || contador) → 1024 bytes
        let mut shake = Shake256::default();
        shake.update(&t);
        shake.update(&self.block_counter.to_be_bytes());

        let mut reader = shake.finalize_xof();
        let mut keystream = [0u8; KS_BLOCK];
        xof_read_exact(&mut reader, &mut keystream);

        self.block_counter = self.block_counter.wrapping_add(1);
        keystream
    }

    /// Gera um arquivo pseudoaleatório de tamanho arbitrário.
    pub fn gerar_arquivo_prng(&mut self, filename: &str, size_bytes: u64) -> io::Result<()> {
        let start = Instant::now();

        let file = File::create(filename)?;
        let mut writer = BufWriter::with_capacity(1024 * 1024, file);

        let mut remaining = size_bytes;
        let mut total_written: u64 = 0;

        while remaining > 0 {
            let block = self.keystream_block_1024();
            let take = remaining.min(KS_BLOCK as u64) as usize;
            writer.write_all(&block[..take])?;
            remaining -= take as u64;
            total_written += take as u64;
        }

        writer.flush()?;

        let elapsed = start.elapsed();
        let seconds = elapsed.as_secs_f64();
        let mb = total_written as f64 / (1024.0 * 1024.0);
        let speed = if seconds > 0.0 { mb / seconds } else { 0.0 };

        println!("Generated file: {}", filename);
        println!("Size: {:.2} MB", mb);
        println!("Total time: {:.3} s", seconds);
        println!("Generation rate: {:.2} MB/s", speed);

        Ok(())
    }

    /// XOR do keystream no buffer inteiro (cifra/decifra).
    /// Implementado corretamente usando 100% do keystream (sem truncar para 64 bytes).
    pub fn apply_keystream_in_place(&mut self, data: &mut [u8]) {
        let mut ks = Keystream1024::new(self);
        ks.xor_in_place(self, data);
    }

    /// Cifra um arquivo para "<arquivo>.enc" com:
    /// - primeiros 64 bytes do arquivo cifrado: TAG = HMAC-SHA3-512(plaintext)
    /// - ciphertext: XOR keystream
    /// A chave do HMAC é derivada dos primeiros 64 bytes do keystream (reservados).
    pub fn cifrar_arquivo(
        password: &[u8],
        salt: &[u8],
        nonce: &[u8],
        input_filename: &str,
    ) -> Result<(), FileCryptoError> {
        let output_filename = format!("{input_filename}.enc");

        let mut cipher = RotorCipher::new(password, salt, nonce)?;
        let mut ks = Keystream1024::new(&mut cipher);

        // Reserva 64 bytes do keystream para MAC key (não usa para cifra)
        let mut mac_key = [0u8; 64];
        ks.take(&mut cipher, &mut mac_key);

        let mut mac = HmacSha3_512::new_from_slice(&mac_key).expect("HMAC accepts any key size");
        mac_key.zeroize();

        let in_file = File::open(input_filename)?;
        let mut reader = BufReader::with_capacity(1024 * 1024, in_file);

        // Abrimos File base para seek no início depois
        let mut out_file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&output_filename)?;

        // placeholder 64 bytes
        out_file.write_all(&[0u8; 64])?;

        // writer via clone
        let out_clone = out_file.try_clone()?;
        let mut writer = BufWriter::with_capacity(1024 * 1024, out_clone);

        let mut buf = vec![0u8; 1024 * 1024];
        loop {
            let n = reader.read(&mut buf)?;
            if n == 0 {
                break;
            }

            // HMAC sobre plaintext
            Mac::update(&mut mac, &buf[..n]);

            // cifra in-place (sem alocação)
            ks.xor_in_place(&mut cipher, &mut buf[..n]);
            writer.write_all(&buf[..n])?;
        }

        writer.flush()?;
        drop(writer);

        let tag = mac.finalize().into_bytes();
        out_file.seek(SeekFrom::Start(0))?;
        out_file.write_all(&tag)?;
        out_file.flush()?;

        Ok(())
    }

    /// Decifra um arquivo cifrado no formato:
    /// [64 bytes TAG][ciphertext...]
    /// e grava "<arquivo>.dec".
    pub fn decifrar_arquivo(
        password: &[u8],
        salt: &[u8],
        nonce: &[u8],
        input_cipher_filename: &str,
    ) -> Result<(), FileCryptoError> {
        let output_filename = format!("{input_cipher_filename}.dec");

        let in_file = File::open(input_cipher_filename)?;
        let mut reader = BufReader::with_capacity(1024 * 1024, in_file);

        let mut expected_tag = [0u8; 64];
        reader
            .read_exact(&mut expected_tag)
            .map_err(|_| FileCryptoError::InvalidCiphertextFormat)?;

        let mut cipher = RotorCipher::new(password, salt, nonce)?;
        let mut ks = Keystream1024::new(&mut cipher);

        // Reserva 64 bytes do keystream para MAC key
        let mut mac_key = [0u8; 64];
        ks.take(&mut cipher, &mut mac_key);

        let mut mac = HmacSha3_512::new_from_slice(&mac_key).expect("HMAC accepts any key size");
        mac_key.zeroize();

        let out_file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&output_filename)?;
        let mut writer = BufWriter::with_capacity(1024 * 1024, out_file);

        let mut buf = vec![0u8; 1024 * 1024];
        loop {
            let n = reader.read(&mut buf)?;
            if n == 0 {
                break;
            }

            // decifra in-place
            ks.xor_in_place(&mut cipher, &mut buf[..n]);

            // HMAC sobre plaintext
            Mac::update(&mut mac, &buf[..n]);

            writer.write_all(&buf[..n])?;
        }

        writer.flush()?;

        let got_tag = mac.finalize().into_bytes();
        if !ct_eq(&expected_tag, &got_tag) {
            drop(writer);
            let _ = std::fs::remove_file(&output_filename);
            return Err(FileCryptoError::IntegrityMismatch);
        }

        Ok(())
    }
}

/// Leitor de keystream em blocos de 1024 bytes (sem alocações no XOR).
struct Keystream1024 {
    buf: [u8; KS_BLOCK],
    pos: usize,
}

impl Keystream1024 {
    fn new(cipher: &mut RotorCipher) -> Self {
        let buf = cipher.keystream_block_1024();
        Self { buf, pos: 0 }
    }

    /// Extrai bytes do keystream para `out` (sem alocação).
    fn take(&mut self, cipher: &mut RotorCipher, out: &mut [u8]) {
        let mut written = 0usize;
        while written < out.len() {
            if self.pos == KS_BLOCK {
                self.buf = cipher.keystream_block_1024();
                self.pos = 0;
            }
            let avail = KS_BLOCK - self.pos;
            let need = out.len() - written;
            let take = avail.min(need);

            out[written..written + take].copy_from_slice(&self.buf[self.pos..self.pos + take]);

            self.pos += take;
            written += take;
        }
    }

    /// XOR in-place usando SIMD XOR (via `xor_inplace`) sem buffer temporário.
    fn xor_in_place(&mut self, cipher: &mut RotorCipher, data: &mut [u8]) {
        let mut offset = 0usize;
        while offset < data.len() {
            if self.pos == KS_BLOCK {
                self.buf = cipher.keystream_block_1024();
                self.pos = 0;
            }

            let avail = KS_BLOCK - self.pos;
            let need = data.len() - offset;
            let take = avail.min(need);

            xor_inplace(
                &mut data[offset..offset + take],
                &self.buf[self.pos..self.pos + take],
            );

            self.pos += take;
            offset += take;
        }
    }
}

/* =========================================================
 * Salt/nonce aleatórios
 * ========================================================= */

pub fn random_salt_nonce() -> ([u8; 16], [u8; 16]) {
    let mut salt = [0u8; 16];
    let mut nonce = [0u8; 16];
    let mut rng = rand::thread_rng();
    rng.fill_bytes(&mut salt);
    rng.fill_bytes(&mut nonce);
    (salt, nonce)
}

/* =========================================================
 * SIMD XOR
 * ========================================================= */

fn xor_inplace(dst: &mut [u8], src: &[u8]) {
    assert_eq!(dst.len(), src.len());

    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") {
            unsafe { xor_inplace_avx2(dst, src) };
            return;
        }
        if std::is_x86_feature_detected!("sse2") {
            unsafe { xor_inplace_sse2(dst, src) };
            return;
        }
    }

    for (d, s) in dst.iter_mut().zip(src.iter()) {
        *d ^= *s;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn xor_inplace_sse2(dst: &mut [u8], src: &[u8]) {
    use std::arch::x86_64::*;
    let mut i = 0usize;
    let n = dst.len();

    while i + 16 <= n {
        let a = unsafe { _mm_loadu_si128(dst.as_ptr().add(i) as *const __m128i) };
        let b = unsafe { _mm_loadu_si128(src.as_ptr().add(i) as *const __m128i) };
        let x = _mm_xor_si128(a, b);
        unsafe { _mm_storeu_si128(dst.as_mut_ptr().add(i) as *mut __m128i, x) };
        i += 16;
    }

    while i < n {
        dst[i] ^= src[i];
        i += 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn xor_inplace_avx2(dst: &mut [u8], src: &[u8]) {
    use std::arch::x86_64::*;
    let mut i = 0usize;
    let n = dst.len();

    while i + 32 <= n {
        let a = unsafe { _mm256_loadu_si256(dst.as_ptr().add(i) as *const __m256i) };
        let b = unsafe { _mm256_loadu_si256(src.as_ptr().add(i) as *const __m256i) };
        let x = _mm256_xor_si256(a, b);
        unsafe { _mm256_storeu_si256(dst.as_mut_ptr().add(i) as *mut __m256i, x) };
        i += 32;
    }

    while i < n {
        dst[i] ^= src[i];
        i += 1;
    }
}

/* =========================================================
 * Exemplo
 * ========================================================= */

fn main() -> Result<(), RotorError> {
    let password = b"senha forte aqui";
    let (salt, nonce) = random_salt_nonce();

    let mut cipher = RotorCipher::new(password, &salt, &nonce)?;

    let mut msg = b"Exemplo de texto a cifrar com RotorCipher".to_vec();
    let original = msg.clone();

    cipher.apply_keystream_in_place(&mut msg); // encrypt

    // Para decifrar: reinicializa com mesmos salt/nonce/password
    let mut cipher2 = RotorCipher::new(password, &salt, &nonce)?;
    cipher2.apply_keystream_in_place(&mut msg); // decrypt

    assert_eq!(msg, original);
    Ok(())
}

/* =========================================================
 * Testes (mantidos)
 * ========================================================= */

#[cfg(test)]
mod tests {
    // use num_bigint::BigInt;

    use super::*;

    #[test]
    fn roundtrip() {
        let password = b"pw";
        let salt = [7u8; 16];
        let nonce = [9u8; 16];

        let mut c1 = RotorCipher::new(password, &salt, &nonce).unwrap();
        let mut data = vec![0u8; 10_000];
        for i in 0..data.len() {
            data[i] = (i * 31 % 256) as u8;
        }
        let original = data.clone();
        c1.apply_keystream_in_place(&mut data);

        let mut c2 = RotorCipher::new(password, &salt, &nonce).unwrap();
        c2.apply_keystream_in_place(&mut data);
        assert_eq!(data, original);
    }

    #[test]
    fn test_generate_file() -> Result<(), Box<dyn std::error::Error>> {
        let password = b"9";
        let salt = [1u8; 16];
        let nonce = [2u8; 16];

        let mut cipher = RotorCipher::new(password, &salt, &nonce)?;
        cipher.gerar_arquivo_prng("saida_prng.bin", 1 * 1024 * 1024 * 1024)?;

        Ok(())
    }

    #[test]
    fn test_encrypt_decrypt_file_roundtrip() -> Result<(), Box<dyn std::error::Error>> {
        use std::fs;
        use std::io::Write;

        let password = b"senha forte aqui";
        let salt = [3u8; 16];
        let nonce = [4u8; 16];

        let input_file = "./saida_prng.bin";
        let enc_file = format!("{input_file}.enc");
        let dec_file = format!("{enc_file}.dec");

        /* ---------------------------------------------------------
         * 1) Cria arquivo de entrada (plaintext)
         * --------------------------------------------------------- */
        {
            let mut f = File::create(input_file)?;
            // Conteúdo determinístico + não trivial
            let tam = (100 * 1024 * 1024) / 16; // 100 MB
            for _i in 0..tam {
                f.write_all(&0u128.to_be_bytes())?;
            }
            f.flush()?;
        }

        /* ---------------------------------------------------------
         * 2) Cifra o arquivo
         * --------------------------------------------------------- */
        let start = Instant::now();
        RotorCipher::cifrar_arquivo(password, &salt, &nonce, input_file)?;
        let elapsed = start.elapsed();
        println!("Encryption time.: {:.3} s", elapsed.as_secs_f64());

        assert!(fs::metadata(&enc_file)?.len() > 64); // pelo menos TAG + dados

        /* ---------------------------------------------------------
         * 3) Decifra o arquivo
         * --------------------------------------------------------- */
        let start = Instant::now();
        RotorCipher::decifrar_arquivo(password, &salt, &nonce, &enc_file)?;
        let elapsed = start.elapsed();
        println!("Decoding time...: {:.3} s", elapsed.as_secs_f64());

        /* ---------------------------------------------------------
         * 4) Verifica integridade (byte a byte)
         * --------------------------------------------------------- */
        let original = fs::read(input_file)?;
        let decrypted = fs::read(&dec_file)?;

        assert_eq!(
            original.len(),
            decrypted.len(),
            "Tamanhos diferentes após decifragem"
        );
        assert_eq!(
            original, decrypted,
            "Conteúdo decifrado não bate com o original"
        );

        /* ---------------------------------------------------------
         * 5) Limpeza
         * --------------------------------------------------------- */
        // let _ = fs::remove_file(input_file);
        // let _ = fs::remove_file(enc_file);
        // let _ = fs::remove_file(dec_file);

        Ok(())
    }
}
