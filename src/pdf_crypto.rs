use std::collections::HashMap;
use std::fmt;

use aes::Aes128;
use cbc::Decryptor;
use cipher::block_padding::Pkcs7;
use cipher::{BlockDecryptMut, KeyIvInit};
use md5::{Digest, Md5};
use rc4::{KeyInit, Rc4, StreamCipher};

use crate::model::Object;

const PASSWORD_PADDING: [u8; 32] = [
    0x28, 0xBF, 0x4E, 0x5E, 0x4E, 0x75, 0x8A, 0x41, 0x64, 0x00, 0x4E, 0x56, 0xFF, 0xFA, 0x01, 0x08,
    0x2E, 0x2E, 0x00, 0xB6, 0xD0, 0x68, 0x3E, 0x80, 0x2F, 0x0C, 0xA9, 0xFE, 0x64, 0x53, 0x69, 0x7A,
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CryptMethod {
    Identity,
    Rc4,
    AesV2,
}

#[derive(Debug)]
pub(crate) enum PdfCryptoError {
    InvalidPassword,
    Unsupported(String),
    Malformed(String),
}

impl fmt::Display for PdfCryptoError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PdfCryptoError::InvalidPassword => {
                write!(f, "invalid password for encrypted PDF")
            }
            PdfCryptoError::Unsupported(msg) => write!(f, "unsupported encryption: {}", msg),
            PdfCryptoError::Malformed(msg) => write!(f, "malformed encryption dictionary: {}", msg),
        }
    }
}

impl std::error::Error for PdfCryptoError {}

#[derive(Debug, Clone)]
pub(crate) struct PdfEncryption {
    encrypt_ref: Option<(u32, u16)>,
    revision: i32,
    key_len: usize,
    o: Vec<u8>,
    u: Vec<u8>,
    permissions: i32,
    file_id: Vec<u8>,
    encrypt_metadata: bool,
    stream_method: CryptMethod,
    string_method: CryptMethod,
}

impl PdfEncryption {
    pub(crate) fn from_dict(
        encrypt_dict: &HashMap<String, Object>,
        encrypt_ref: Option<(u32, u16)>,
        file_id: Vec<u8>,
    ) -> Result<Self, PdfCryptoError> {
        let filter = encrypt_dict
            .get("Filter")
            .and_then(|o| o.as_name())
            .ok_or_else(|| PdfCryptoError::Malformed("missing /Filter".to_string()))?;
        if filter != "Standard" {
            return Err(PdfCryptoError::Unsupported(format!(
                "security handler {}",
                filter
            )));
        }

        let revision = encrypt_dict
            .get("R")
            .and_then(|o| o.as_i64())
            .ok_or_else(|| PdfCryptoError::Malformed("missing /R".to_string()))?
            as i32;
        if !matches!(revision, 2 | 3 | 4) {
            return Err(PdfCryptoError::Unsupported(format!(
                "Standard handler revision R={}",
                revision
            )));
        }

        let version = encrypt_dict.get("V").and_then(|o| o.as_i64()).unwrap_or(0) as i32;
        if !matches!(version, 1 | 2 | 4) {
            return Err(PdfCryptoError::Unsupported(format!(
                "encryption version V={}",
                version
            )));
        }

        let o = match encrypt_dict.get("O") {
            Some(Object::String(v)) => v.clone(),
            _ => return Err(PdfCryptoError::Malformed("missing /O".to_string())),
        };
        let u = match encrypt_dict.get("U") {
            Some(Object::String(v)) => v.clone(),
            _ => return Err(PdfCryptoError::Malformed("missing /U".to_string())),
        };
        if o.len() < 32 || u.len() < 16 {
            return Err(PdfCryptoError::Malformed(
                "unexpected /O or /U length".to_string(),
            ));
        }

        let permissions = encrypt_dict
            .get("P")
            .and_then(|o| o.as_i64())
            .ok_or_else(|| PdfCryptoError::Malformed("missing /P".to_string()))?;
        let permissions = i32::try_from(permissions)
            .map_err(|_| PdfCryptoError::Malformed("invalid /P value".to_string()))?;

        if file_id.is_empty() {
            return Err(PdfCryptoError::Malformed(
                "missing file identifier (/ID)".to_string(),
            ));
        }

        let key_len_bits = encrypt_dict
            .get("Length")
            .and_then(|o| o.as_i64())
            .unwrap_or(if revision == 2 { 40 } else { 128 });
        if key_len_bits <= 0 || key_len_bits % 8 != 0 {
            return Err(PdfCryptoError::Malformed("invalid /Length".to_string()));
        }
        let key_len = usize::try_from(key_len_bits / 8)
            .map_err(|_| PdfCryptoError::Malformed("invalid key length".to_string()))?;
        if key_len == 0 || key_len > 16 {
            return Err(PdfCryptoError::Malformed(
                "unsupported key length for R2-R4".to_string(),
            ));
        }
        let key_len = if version == 1 { 5 } else { key_len };

        let encrypt_metadata = encrypt_dict
            .get("EncryptMetadata")
            .and_then(|o| match o {
                Object::Boolean(v) => Some(*v),
                _ => None,
            })
            .unwrap_or(true);

        let (stream_method, string_method) = if version == 4 {
            let cf = encrypt_dict.get("CF").and_then(|o| o.as_dict());
            let stmf = encrypt_dict
                .get("StmF")
                .and_then(|o| o.as_name())
                .unwrap_or("Identity");
            let strf = encrypt_dict
                .get("StrF")
                .and_then(|o| o.as_name())
                .unwrap_or("Identity");
            (
                resolve_crypt_method(stmf, cf)?,
                resolve_crypt_method(strf, cf)?,
            )
        } else {
            (CryptMethod::Rc4, CryptMethod::Rc4)
        };

        if matches!(stream_method, CryptMethod::AesV2)
            || matches!(string_method, CryptMethod::AesV2)
        {
            if key_len != 16 {
                return Err(PdfCryptoError::Malformed(
                    "AESV2 requires 128-bit key length".to_string(),
                ));
            }
        }

        Ok(Self {
            encrypt_ref,
            revision,
            key_len,
            o,
            u,
            permissions,
            file_id,
            encrypt_metadata,
            stream_method,
            string_method,
        })
    }

    pub(crate) fn encrypt_ref(&self) -> Option<(u32, u16)> {
        self.encrypt_ref
    }

    pub(crate) fn encrypt_metadata(&self) -> bool {
        self.encrypt_metadata
    }

    pub(crate) fn stream_method(&self) -> CryptMethod {
        self.stream_method
    }

    pub(crate) fn string_method(&self) -> CryptMethod {
        self.string_method
    }

    pub(crate) fn authenticate(&self, password: Option<&[u8]>) -> Result<Vec<u8>, PdfCryptoError> {
        let password = password.unwrap_or(&[]);

        let direct_key = self.derive_file_key(password);
        if self.validates_user_key(&direct_key) {
            return Ok(direct_key);
        }

        let recovered_user_password = self.owner_password_to_user_password(password)?;
        let owner_key = self.derive_file_key(&recovered_user_password);
        if self.validates_user_key(&owner_key) {
            return Ok(owner_key);
        }

        Err(PdfCryptoError::InvalidPassword)
    }

    pub(crate) fn decrypt_bytes(
        &self,
        method: CryptMethod,
        file_key: &[u8],
        obj_num: u32,
        gen_num: u16,
        data: &[u8],
    ) -> Result<Vec<u8>, PdfCryptoError> {
        match method {
            CryptMethod::Identity => Ok(data.to_vec()),
            CryptMethod::Rc4 => {
                let mut out = data.to_vec();
                let obj_key = self.object_key(file_key, obj_num, gen_num, false);
                rc4_apply(&obj_key, &mut out)?;
                Ok(out)
            }
            CryptMethod::AesV2 => {
                if data.len() < 16 {
                    return Err(PdfCryptoError::Malformed(
                        "AESV2 payload is missing IV".to_string(),
                    ));
                }
                let obj_key = self.object_key(file_key, obj_num, gen_num, true);
                if obj_key.len() != 16 {
                    return Err(PdfCryptoError::Malformed(
                        "AESV2 object key must be 16 bytes".to_string(),
                    ));
                }
                let iv = &data[..16];
                let mut encrypted = data[16..].to_vec();
                let decryptor =
                    Decryptor::<Aes128>::new_from_slices(&obj_key, iv).map_err(|_| {
                        PdfCryptoError::Malformed("invalid AES key/iv length".to_string())
                    })?;
                let plaintext = decryptor
                    .decrypt_padded_mut::<Pkcs7>(&mut encrypted)
                    .map_err(|_| PdfCryptoError::Malformed("invalid AESV2 padding".to_string()))?;
                Ok(plaintext.to_vec())
            }
        }
    }

    fn derive_file_key(&self, password: &[u8]) -> Vec<u8> {
        let mut input = Vec::with_capacity(32 + self.o.len() + 4 + self.file_id.len() + 4);
        input.extend_from_slice(&pad_password(password));
        input.extend_from_slice(&self.o);
        input.extend_from_slice(&self.permissions.to_le_bytes());
        input.extend_from_slice(&self.file_id);
        if self.revision >= 4 && !self.encrypt_metadata {
            input.extend_from_slice(&[0xFF, 0xFF, 0xFF, 0xFF]);
        }

        let mut digest = md5_sum(&input).to_vec();
        if self.revision >= 3 {
            for _ in 0..50 {
                digest = md5_sum(&digest[..self.key_len]).to_vec();
            }
        }
        digest[..self.key_len].to_vec()
    }

    fn validates_user_key(&self, file_key: &[u8]) -> bool {
        if self.revision == 2 {
            if self.u.len() < 32 {
                return false;
            }
            let mut value = PASSWORD_PADDING.to_vec();
            if rc4_apply(file_key, &mut value).is_err() {
                return false;
            }
            return value == self.u[..32];
        }

        if self.u.len() < 16 {
            return false;
        }
        let mut input = Vec::with_capacity(32 + self.file_id.len());
        input.extend_from_slice(&PASSWORD_PADDING);
        input.extend_from_slice(&self.file_id);
        let digest = md5_sum(&input);

        let mut value = digest.to_vec();
        if rc4_apply(file_key, &mut value).is_err() {
            return false;
        }
        for i in 1..=19u8 {
            let key = xor_key(file_key, i);
            if rc4_apply(&key, &mut value).is_err() {
                return false;
            }
        }
        value[..16] == self.u[..16]
    }

    fn owner_password_to_user_password(
        &self,
        owner_password: &[u8],
    ) -> Result<Vec<u8>, PdfCryptoError> {
        let mut digest = md5_sum(&pad_password(owner_password)).to_vec();
        if self.revision >= 3 {
            for _ in 0..50 {
                digest = md5_sum(&digest[..self.key_len]).to_vec();
            }
        }
        let owner_key = &digest[..self.key_len];

        let mut value = self.o.clone();
        if self.revision == 2 {
            rc4_apply(owner_key, &mut value)?;
            return Ok(value);
        }

        for i in (0..=19u8).rev() {
            let key = xor_key(owner_key, i);
            rc4_apply(&key, &mut value)?;
        }
        Ok(value)
    }

    fn object_key(&self, file_key: &[u8], obj_num: u32, gen_num: u16, aes_salt: bool) -> Vec<u8> {
        let mut material = Vec::with_capacity(file_key.len() + 9);
        material.extend_from_slice(file_key);
        material.push((obj_num & 0xFF) as u8);
        material.push(((obj_num >> 8) & 0xFF) as u8);
        material.push(((obj_num >> 16) & 0xFF) as u8);
        material.push((gen_num & 0xFF) as u8);
        material.push(((gen_num >> 8) & 0xFF) as u8);
        if aes_salt {
            material.extend_from_slice(b"sAlT");
        }
        let digest = md5_sum(&material);
        let key_len = (file_key.len() + 5).min(16);
        digest[..key_len].to_vec()
    }
}

fn resolve_crypt_method(
    filter_name: &str,
    cf_dict: Option<&HashMap<String, Object>>,
) -> Result<CryptMethod, PdfCryptoError> {
    if filter_name == "Identity" {
        return Ok(CryptMethod::Identity);
    }
    let cf_dict =
        cf_dict.ok_or_else(|| PdfCryptoError::Malformed("missing /CF dictionary".to_string()))?;
    let cf = cf_dict
        .get(filter_name)
        .and_then(|o| o.as_dict())
        .ok_or_else(|| PdfCryptoError::Malformed(format!("missing /CF entry {}", filter_name)))?;
    let cfm = cf.get("CFM").and_then(|o| o.as_name()).unwrap_or("None");
    match cfm {
        "None" => Ok(CryptMethod::Identity),
        "V2" => Ok(CryptMethod::Rc4),
        "AESV2" => Ok(CryptMethod::AesV2),
        other => Err(PdfCryptoError::Unsupported(format!(
            "crypt filter method {}",
            other
        ))),
    }
}

fn pad_password(password: &[u8]) -> [u8; 32] {
    let mut out = PASSWORD_PADDING;
    let copy_len = password.len().min(32);
    out[..copy_len].copy_from_slice(&password[..copy_len]);
    out
}

fn md5_sum(data: &[u8]) -> [u8; 16] {
    let mut hasher = Md5::new();
    hasher.update(data);
    let digest = hasher.finalize();
    let mut out = [0u8; 16];
    out.copy_from_slice(&digest);
    out
}

fn xor_key(key: &[u8], value: u8) -> Vec<u8> {
    key.iter().map(|b| b ^ value).collect()
}

fn rc4_apply(key: &[u8], data: &mut [u8]) -> Result<(), PdfCryptoError> {
    if key.is_empty() || key.len() > 16 {
        return Err(PdfCryptoError::Malformed(format!(
            "unsupported RC4 key length {}",
            key.len()
        )));
    }
    match key.len() {
        1 => rc4_apply_sized::<rc4::consts::U1>(key, data),
        2 => rc4_apply_sized::<rc4::consts::U2>(key, data),
        3 => rc4_apply_sized::<rc4::consts::U3>(key, data),
        4 => rc4_apply_sized::<rc4::consts::U4>(key, data),
        5 => rc4_apply_sized::<rc4::consts::U5>(key, data),
        6 => rc4_apply_sized::<rc4::consts::U6>(key, data),
        7 => rc4_apply_sized::<rc4::consts::U7>(key, data),
        8 => rc4_apply_sized::<rc4::consts::U8>(key, data),
        9 => rc4_apply_sized::<rc4::consts::U9>(key, data),
        10 => rc4_apply_sized::<rc4::consts::U10>(key, data),
        11 => rc4_apply_sized::<rc4::consts::U11>(key, data),
        12 => rc4_apply_sized::<rc4::consts::U12>(key, data),
        13 => rc4_apply_sized::<rc4::consts::U13>(key, data),
        14 => rc4_apply_sized::<rc4::consts::U14>(key, data),
        15 => rc4_apply_sized::<rc4::consts::U15>(key, data),
        16 => rc4_apply_sized::<rc4::consts::U16>(key, data),
        _ => Err(PdfCryptoError::Malformed(
            "unsupported RC4 key length".to_string(),
        )),
    }
}

fn rc4_apply_sized<K>(key: &[u8], data: &mut [u8]) -> Result<(), PdfCryptoError>
where
    Rc4<K>: KeyInit + StreamCipher,
{
    let mut cipher = Rc4::<K>::new_from_slice(key)
        .map_err(|_| PdfCryptoError::Malformed("invalid RC4 key".to_string()))?;
    cipher.apply_keystream(data);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use cbc::Encryptor;
    use cipher::BlockEncryptMut;

    #[test]
    fn rc4_roundtrip() {
        let key = b"abcde";
        let plain = b"hello world";
        let mut data = plain.to_vec();
        rc4_apply(key, &mut data).expect("encrypt");
        assert_ne!(data, plain);
        rc4_apply(key, &mut data).expect("decrypt");
        assert_eq!(data, plain);
    }

    #[test]
    fn aesv2_roundtrip_for_object_bytes() {
        let encryption = PdfEncryption {
            encrypt_ref: None,
            revision: 4,
            key_len: 16,
            o: vec![0; 32],
            u: vec![0; 32],
            permissions: -4,
            file_id: vec![1, 2, 3, 4, 5, 6, 7, 8],
            encrypt_metadata: true,
            stream_method: CryptMethod::AesV2,
            string_method: CryptMethod::AesV2,
        };
        let file_key: Vec<u8> = (0u8..16).collect();
        let obj_num = 42u32;
        let gen_num = 0u16;
        let obj_key = encryption.object_key(&file_key, obj_num, gen_num, true);
        assert_eq!(obj_key.len(), 16);

        let plaintext = b"BT /F1 12 Tf (Hello) Tj ET";
        let iv = [7u8; 16];
        let mut padded = vec![0u8; plaintext.len() + 16];
        padded[..plaintext.len()].copy_from_slice(plaintext);
        let encryptor = Encryptor::<Aes128>::new_from_slices(&obj_key, &iv).expect("cipher");
        let ciphertext = encryptor
            .encrypt_padded_mut::<Pkcs7>(&mut padded, plaintext.len())
            .expect("padded");

        let mut payload = iv.to_vec();
        payload.extend_from_slice(ciphertext);

        let decrypted = encryption
            .decrypt_bytes(CryptMethod::AesV2, &file_key, obj_num, gen_num, &payload)
            .expect("decrypt");
        assert_eq!(decrypted, plaintext);
    }
}
