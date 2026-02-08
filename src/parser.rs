use std::collections::{HashMap, VecDeque};
use std::fmt;

use crate::model::Object;
use crate::pdf_crypto::{CryptMethod, PdfCryptoError, PdfEncryption};
use crate::tokenizer::{Lexer, Token};

#[derive(Debug)]
pub enum ParseError {
    InvalidPassword,
    UnsupportedEncryption(String),
    MalformedEncryption(String),
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ParseError::InvalidPassword => write!(f, "invalid password for encrypted PDF"),
            ParseError::UnsupportedEncryption(msg) => {
                write!(f, "unsupported encryption: {}", msg)
            }
            ParseError::MalformedEncryption(msg) => {
                write!(f, "malformed encryption data: {}", msg)
            }
        }
    }
}

impl std::error::Error for ParseError {}

#[derive(Debug)]
pub struct PdfDoc {
    pub objects: HashMap<(u32, u16), Object>,
    pub trailer: Option<Object>,
}

impl PdfDoc {
    pub fn get(&self, obj_num: u32, gen_num: u16) -> Option<&Object> {
        self.objects.get(&(obj_num, gen_num))
    }

    pub fn resolve<'a>(&'a self, obj: &'a Object) -> &'a Object {
        match obj {
            Object::Reference { obj_num, gen_num } => {
                self.objects.get(&(*obj_num, *gen_num)).unwrap_or(obj)
            }
            _ => obj,
        }
    }

    pub fn resolve_owned(&self, obj: &Object) -> Option<Object> {
        match obj {
            Object::Reference { obj_num, gen_num } => {
                self.objects.get(&(*obj_num, *gen_num)).cloned()
            }
            _ => Some(obj.clone()),
        }
    }

    pub fn expand_object_streams(&mut self) {
        // Object streams can hold indirect objects that are needed later in parsing/extraction.
        let mut object_stream_keys: Vec<(u32, u16)> = Vec::new();
        for (key, obj) in &self.objects {
            if let Object::Stream { dict, .. } = obj {
                let ty = dict.get("Type").and_then(|v| v.as_name());
                if ty == Some("ObjStm") {
                    object_stream_keys.push(*key);
                }
            }
        }
        object_stream_keys.sort_unstable();

        let mut updates: HashMap<(u32, u16), Object> = HashMap::new();
        for key in object_stream_keys {
            let Some(Object::Stream { dict, data }) = self.objects.get(&key) else {
                continue;
            };
            let n = dict.get("N").and_then(|v| v.as_i64()).unwrap_or(0) as usize;
            let first = dict.get("First").and_then(|v| v.as_i64()).unwrap_or(0) as usize;
            if n == 0 || first == 0 {
                continue;
            }
            let stream_data = decode_stream_data(&dict, &data);
            if stream_data.len() < first {
                continue;
            }
            let mut header_lexer = Lexer::new(&stream_data);
            let mut entries = Vec::with_capacity(n);
            for _ in 0..n {
                let obj_num = match header_lexer.next_token() {
                    Some(Token::Integer(v)) => v as u32,
                    _ => break,
                };
                let offset = match header_lexer.next_token() {
                    Some(Token::Integer(v)) => v as usize,
                    _ => break,
                };
                entries.push((obj_num, offset));
            }
            for (obj_num, offset) in entries {
                let key = (obj_num, 0);
                let pos = first + offset;
                if pos >= stream_data.len()
                    || self.objects.contains_key(&key)
                    || updates.contains_key(&key)
                {
                    continue;
                }
                if let Some(obj) = parse_object_at(&stream_data, pos) {
                    // In incremental PDFs, direct objects can supersede stale ObjStm entries.
                    // Fill gaps from object streams, but never overwrite already parsed objects.
                    updates.insert(key, obj);
                }
            }
        }

        for (key, obj) in updates {
            self.objects.insert(key, obj);
        }
    }
}

pub struct Parser<'a> {
    lexer: Lexer<'a>,
    buffer: VecDeque<Token>,
    length_ref_cache: HashMap<(u32, u16), Option<usize>>,
    integer_object_header_index: Option<HashMap<(u32, u16), Vec<usize>>>,
}

impl<'a> Parser<'a> {
    pub fn new(lexer: Lexer<'a>) -> Self {
        Self {
            lexer,
            buffer: VecDeque::new(),
            length_ref_cache: HashMap::new(),
            integer_object_header_index: None,
        }
    }

    pub fn parse(self) -> Result<PdfDoc, ParseError> {
        self.parse_with_password(None)
    }

    pub fn parse_with_password(mut self, password: Option<&[u8]>) -> Result<PdfDoc, ParseError> {
        let mut doc = self.parse_raw();

        if let Some(encryption) = build_encryption(&doc)? {
            let file_key = encryption
                .authenticate(password)
                .map_err(map_crypto_error)?;
            decrypt_document(&mut doc, &encryption, &file_key)?;
        }

        doc.expand_object_streams();
        Ok(doc)
    }

    fn parse_raw(&mut self) -> PdfDoc {
        let mut objects: HashMap<(u32, u16), Object> = HashMap::new();
        let mut trailer: Option<Object> = None;

        while let Some(token) = self.next_token() {
            match token {
                Token::Integer(obj_num) => {
                    let gen_num = match self.peek_token_ref() {
                        Some(Token::Integer(v)) => Some(*v),
                        _ => None,
                    };
                    let has_obj_keyword =
                        matches!(self.peek_token_n_ref(2), Some(Token::Keyword(kw)) if kw == "obj");
                    if let (Some(gen_num), true) = (gen_num, has_obj_keyword) {
                        let _ = self.next_token();
                        let _ = self.next_token();
                        if let Some(obj) = self.parse_indirect_object(&objects) {
                            if trailer.is_none() {
                                if let Object::Stream { dict, .. } = &obj {
                                    if dict.get("Type").and_then(|v| v.as_name()) == Some("XRef") {
                                        trailer = Some(Object::Dictionary(dict.clone()));
                                    }
                                }
                            }
                            objects.insert((obj_num as u32, gen_num as u16), obj);
                        }
                    }
                }
                Token::Keyword(ref kw) if kw == "trailer" => {
                    if let Some(obj) = self.parse_object(&objects) {
                        trailer = Some(obj);
                    }
                }
                _ => {}
            }
        }

        PdfDoc { objects, trailer }
    }

    fn parse_indirect_object(&mut self, objects: &HashMap<(u32, u16), Object>) -> Option<Object> {
        let obj = self.parse_object(objects)?;
        loop {
            let (has_token, at_endobj) = match self.peek_token_ref() {
                Some(Token::Keyword(kw)) if kw == "endobj" => (true, true),
                Some(_) => (true, false),
                None => (false, false),
            };
            if at_endobj {
                let _ = self.next_token();
                break;
            }
            if has_token {
                let _ = self.next_token();
            } else {
                break;
            }
        }
        Some(obj)
    }

    pub(crate) fn parse_object(&mut self, objects: &HashMap<(u32, u16), Object>) -> Option<Object> {
        let token = self.next_token()?;
        match token {
            Token::Null => Some(Object::Null),
            Token::Boolean(v) => Some(Object::Boolean(v)),
            Token::Integer(v) => self.parse_number_or_ref(v),
            Token::Real(v) => Some(Object::Real(v)),
            Token::String(v) => Some(Object::String(v)),
            Token::HexString(v) => Some(Object::String(v)),
            Token::Name(v) => Some(Object::Name(v)),
            Token::ArrayStart => Some(Object::Array(self.parse_array(objects))),
            Token::DictStart => self.parse_dict_or_stream(objects),
            Token::Keyword(_) => None,
            _ => None,
        }
    }

    fn parse_number_or_ref(&mut self, first: i64) -> Option<Object> {
        let second = match self.peek_token_ref() {
            Some(Token::Integer(v)) => Some(*v),
            _ => None,
        };
        if let Some(second) = second {
            let is_ref = matches!(self.peek_token_n_ref(2), Some(Token::Keyword(kw)) if kw == "R");
            if is_ref {
                let _ = self.next_token();
                let _ = self.next_token();
                return Some(Object::Reference {
                    obj_num: first as u32,
                    gen_num: second as u16,
                });
            }
        }
        Some(Object::Integer(first))
    }

    fn parse_array(&mut self, objects: &HashMap<(u32, u16), Object>) -> Vec<Object> {
        let mut items = Vec::new();
        loop {
            let is_array_end = matches!(self.peek_token_ref(), Some(Token::ArrayEnd));
            if is_array_end {
                let _ = self.next_token();
                break;
            }
            if self.peek_token_ref().is_none() {
                break;
            }
            if let Some(obj) = self.parse_object(objects) {
                items.push(obj);
            } else {
                let _ = self.next_token();
            }
        }
        items
    }

    fn parse_dict_or_stream(&mut self, objects: &HashMap<(u32, u16), Object>) -> Option<Object> {
        let mut dict = HashMap::new();
        loop {
            let is_dict_end = matches!(self.peek_token_ref(), Some(Token::DictEnd));
            if is_dict_end {
                let _ = self.next_token();
                break;
            }
            if self.peek_token_ref().is_none() {
                break;
            }
            let key = match self.next_token() {
                Some(Token::Name(v)) => v,
                _ => break,
            };
            if let Some(value) = self.parse_object(objects) {
                dict.insert(key, value);
            } else {
                break;
            }
        }

        let has_stream_keyword =
            matches!(self.peek_token_ref(), Some(Token::Keyword(kw)) if kw == "stream");
        match has_stream_keyword {
            true => {
                let _ = self.next_token();
                let length = self.stream_length(&dict, objects);
                let stream_start = self.lexer.position();
                let mut data = if let Some(length) = length {
                    self.lexer.consume_stream(length)
                } else {
                    self.lexer.consume_stream_until_endstream()
                };

                if length.is_some() {
                    let has_endstream_keyword = matches!(
                        self.peek_token_ref(),
                        Some(Token::Keyword(kw)) if kw == "endstream"
                    );
                    if !has_endstream_keyword {
                        let at_endobj_keyword = matches!(
                            self.peek_token_ref(),
                            Some(Token::Keyword(kw)) if kw == "endobj"
                        );
                        let near_endstream = self.find_nearby_keyword_offset(b"endstream", 256);
                        let near_endobj = self.find_nearby_keyword_offset(b"endobj", 256);
                        let should_rescan = match (near_endstream, near_endobj) {
                            (Some(endstream_pos), Some(endobj_pos)) => endstream_pos <= endobj_pos,
                            (Some(_), None) => true,
                            (None, Some(_)) => false,
                            (None, None) => true,
                        };
                        if !at_endobj_keyword && should_rescan {
                            self.lexer.set_position(stream_start);
                            data = self.lexer.consume_stream_until_endstream();
                        }
                    }
                }

                let has_endstream_keyword = matches!(
                    self.peek_token_ref(),
                    Some(Token::Keyword(kw)) if kw == "endstream"
                );
                if has_endstream_keyword {
                    let _ = self.next_token();
                }
                Some(Object::Stream { dict, data })
            }
            false => Some(Object::Dictionary(dict)),
        }
    }

    fn next_token(&mut self) -> Option<Token> {
        if let Some(token) = self.buffer.pop_front() {
            return Some(token);
        }
        self.lexer.next_token()
    }

    fn peek_token_ref(&mut self) -> Option<&Token> {
        self.peek_token_n_ref(1)
    }

    fn peek_token_n_ref(&mut self, n: usize) -> Option<&Token> {
        if n == 0 {
            return None;
        }
        self.fill_peek_buffer(n);
        self.buffer.get(n - 1)
    }

    fn fill_peek_buffer(&mut self, n: usize) {
        while self.buffer.len() < n {
            if let Some(token) = self.lexer.next_token() {
                self.buffer.push_back(token);
            } else {
                break;
            }
        }
    }

    fn find_nearby_keyword_offset(&self, keyword: &[u8], window: usize) -> Option<usize> {
        if keyword.is_empty() {
            return None;
        }
        let input = self.lexer.input();
        let start = self.lexer.position();
        if start >= input.len() {
            return None;
        }
        let end = start.saturating_add(window).min(input.len());
        for rel in memchr::memmem::find_iter(&input[start..end], keyword) {
            let marker = start + rel;
            let before_ok =
                marker == 0 || is_pdf_delim_or_whitespace(input[marker.saturating_sub(1)]);
            let after = marker + keyword.len();
            let after_ok = after >= input.len() || is_pdf_delim_or_whitespace(input[after]);
            if before_ok && after_ok {
                return Some(rel);
            }
        }
        None
    }

    fn stream_length(
        &mut self,
        dict: &HashMap<String, Object>,
        objects: &HashMap<(u32, u16), Object>,
    ) -> Option<usize> {
        let length_obj = dict.get("Length")?;

        match length_obj {
            Object::Integer(v) => {
                if let Some(length) = positive_i64_to_usize(*v) {
                    Some(length)
                } else {
                    None
                }
            }
            Object::Reference { .. } => match resolve_length_value(length_obj, objects, 0) {
                Some(v) if positive_i64_to_usize(v).is_some() => positive_i64_to_usize(v),
                Some(_) => None,
                None => {
                    let (ref_obj_num, ref_gen_num) = match length_obj {
                        Object::Reference { obj_num, gen_num } => (*obj_num, *gen_num),
                        _ => unreachable!(),
                    };
                    self.resolve_forward_length_reference(ref_obj_num, ref_gen_num)
                }
            },
            _ => None,
        }
    }

    fn resolve_forward_length_reference(&mut self, obj_num: u32, gen_num: u16) -> Option<usize> {
        let key = (obj_num, gen_num);
        if let Some(cached) = self.length_ref_cache.get(&key) {
            return *cached;
        }

        let resolved = self.resolve_forward_length_reference_uncached(obj_num, gen_num);
        self.length_ref_cache.insert(key, resolved);
        resolved
    }

    fn resolve_forward_length_reference_uncached(
        &mut self,
        obj_num: u32,
        gen_num: u16,
    ) -> Option<usize> {
        if self.integer_object_header_index.is_none() {
            let index = scan_indirect_object_headers(self.lexer.input());
            self.integer_object_header_index = Some(index);
        }
        let offsets = self
            .integer_object_header_index
            .as_ref()
            .and_then(|idx| idx.get(&(obj_num, gen_num)))
            .cloned()?;
        let input = self.lexer.input();
        let cursor = self.lexer.position();

        for &offset in &offsets {
            if offset <= cursor {
                continue;
            }
            if let Some(length) =
                parse_positive_integer_indirect_object_at(input, offset, obj_num, gen_num)
            {
                return Some(length);
            }
        }
        for &offset in &offsets {
            if let Some(length) =
                parse_positive_integer_indirect_object_at(input, offset, obj_num, gen_num)
            {
                return Some(length);
            }
        }
        None
    }
}

fn map_crypto_error(err: PdfCryptoError) -> ParseError {
    match err {
        PdfCryptoError::InvalidPassword => ParseError::InvalidPassword,
        PdfCryptoError::Unsupported(msg) => ParseError::UnsupportedEncryption(msg),
        PdfCryptoError::Malformed(msg) => ParseError::MalformedEncryption(msg),
    }
}

fn build_encryption(doc: &PdfDoc) -> Result<Option<PdfEncryption>, ParseError> {
    let Some(trailer_dict) = doc.trailer.as_ref().and_then(|t| t.as_dict()) else {
        return Ok(None);
    };
    let Some(encrypt_obj) = trailer_dict.get("Encrypt") else {
        return Ok(None);
    };

    let encrypt_ref = match encrypt_obj {
        Object::Reference { obj_num, gen_num } => Some((*obj_num, *gen_num)),
        _ => None,
    };
    let encrypt_dict = doc.resolve(encrypt_obj).as_dict().ok_or_else(|| {
        ParseError::MalformedEncryption("/Encrypt is not a dictionary".to_string())
    })?;
    let file_id = extract_file_id(doc, trailer_dict)?;

    let encryption =
        PdfEncryption::from_dict(encrypt_dict, encrypt_ref, file_id).map_err(map_crypto_error)?;
    Ok(Some(encryption))
}

fn extract_file_id(
    doc: &PdfDoc,
    trailer_dict: &HashMap<String, Object>,
) -> Result<Vec<u8>, ParseError> {
    let id_obj = trailer_dict.get("ID").ok_or_else(|| {
        ParseError::MalformedEncryption("encrypted file is missing /ID".to_string())
    })?;
    let id_array = doc
        .resolve(id_obj)
        .as_array()
        .ok_or_else(|| ParseError::MalformedEncryption("/ID is not an array".to_string()))?;
    let first = id_array
        .first()
        .ok_or_else(|| ParseError::MalformedEncryption("/ID is empty".to_string()))?;
    match doc.resolve(first) {
        Object::String(v) => Ok(v.clone()),
        _ => Err(ParseError::MalformedEncryption(
            "first /ID entry is not a string".to_string(),
        )),
    }
}

fn decrypt_document(
    doc: &mut PdfDoc,
    encryption: &PdfEncryption,
    file_key: &[u8],
) -> Result<(), ParseError> {
    for ((obj_num, gen_num), obj) in doc.objects.iter_mut() {
        if encryption.encrypt_ref() == Some((*obj_num, *gen_num)) {
            continue;
        }
        decrypt_object_value(obj, *obj_num, *gen_num, encryption, file_key)?;
    }
    Ok(())
}

fn decrypt_object_value(
    obj: &mut Object,
    obj_num: u32,
    gen_num: u16,
    encryption: &PdfEncryption,
    file_key: &[u8],
) -> Result<(), ParseError> {
    match obj {
        Object::String(bytes) => {
            match encryption.decrypt_bytes(
                encryption.string_method(),
                file_key,
                obj_num,
                gen_num,
                bytes,
            ) {
                Ok(decrypted) => *bytes = decrypted,
                Err(PdfCryptoError::Malformed(_))
                    if encryption.string_method() == CryptMethod::AesV2 =>
                {
                    // Some incremental updates can contain clear-text strings.
                    // Preserve bytes when AESV2 payload shape is invalid.
                }
                Err(err) => return Err(map_crypto_error(err)),
            }
        }
        Object::Array(items) => {
            for item in items {
                decrypt_object_value(item, obj_num, gen_num, encryption, file_key)?;
            }
        }
        Object::Dictionary(dict) => {
            for value in dict.values_mut() {
                decrypt_object_value(value, obj_num, gen_num, encryption, file_key)?;
            }
        }
        Object::Stream { dict, data } => {
            for value in dict.values_mut() {
                decrypt_object_value(value, obj_num, gen_num, encryption, file_key)?;
            }
            if should_decrypt_stream_data(dict, encryption) {
                match encryption.decrypt_bytes(
                    encryption.stream_method(),
                    file_key,
                    obj_num,
                    gen_num,
                    data,
                ) {
                    Ok(decrypted) => *data = decrypted,
                    Err(PdfCryptoError::Malformed(_))
                        if encryption.stream_method() == CryptMethod::AesV2 =>
                    {
                        // Some encrypted PDFs may carry unencrypted hint streams.
                        // Keep original bytes when AESV2 payload shape is invalid.
                    }
                    Err(err) => return Err(map_crypto_error(err)),
                }
            }
        }
        _ => {}
    }
    Ok(())
}

fn should_decrypt_stream_data(dict: &HashMap<String, Object>, encryption: &PdfEncryption) -> bool {
    if encryption.stream_method() == CryptMethod::Identity {
        return false;
    }
    if dict.get("Type").and_then(|o| o.as_name()) == Some("XRef") {
        return false;
    }
    if !encryption.encrypt_metadata()
        && dict.get("Type").and_then(|o| o.as_name()) == Some("Metadata")
    {
        return false;
    }
    true
}

fn parse_object_at(data: &[u8], offset: usize) -> Option<Object> {
    let mut lexer = Lexer::new(data);
    lexer.set_position(offset);
    let mut parser = Parser::new(lexer);
    let objects = HashMap::new();
    parser.parse_object(&objects)
}

fn resolve_length_value(
    value: &Object,
    objects: &HashMap<(u32, u16), Object>,
    depth: usize,
) -> Option<i64> {
    if depth >= 8 {
        return None;
    }
    match value {
        Object::Integer(v) => Some(*v),
        Object::Reference { obj_num, gen_num } => {
            let next = objects.get(&(*obj_num, *gen_num))?;
            resolve_length_value(next, objects, depth + 1)
        }
        _ => None,
    }
}

fn scan_indirect_object_headers(input: &[u8]) -> HashMap<(u32, u16), Vec<usize>> {
    let mut out: HashMap<(u32, u16), Vec<usize>> = HashMap::new();
    for marker in memchr::memmem::find_iter(input, b" obj") {
        let after_obj = marker + 4;
        if after_obj < input.len() && !is_pdf_delim_or_whitespace(input[after_obj]) {
            continue;
        }

        let gen_end = marker;
        let mut gen_start = gen_end;
        while gen_start > 0 && input[gen_start - 1].is_ascii_digit() {
            gen_start -= 1;
        }
        if gen_start == gen_end {
            continue;
        }

        if gen_start == 0 || !is_pdf_whitespace(input[gen_start - 1]) {
            continue;
        }
        let mut obj_end = gen_start - 1;
        while obj_end > 0 && is_pdf_whitespace(input[obj_end - 1]) {
            obj_end -= 1;
        }

        let mut obj_start = obj_end;
        while obj_start > 0 && input[obj_start - 1].is_ascii_digit() {
            obj_start -= 1;
        }
        if obj_start == obj_end {
            continue;
        }
        if obj_start > 0 && !is_pdf_delim_or_whitespace(input[obj_start - 1]) {
            continue;
        }

        let Some((obj_num, obj_after)) = parse_ascii_u64(input, obj_start) else {
            continue;
        };
        if obj_after != obj_end || obj_num > u32::MAX as u64 {
            continue;
        }

        let Some((gen_num, gen_after)) = parse_ascii_u64(input, gen_start) else {
            continue;
        };
        if gen_after != gen_end || gen_num > u16::MAX as u64 {
            continue;
        }

        out.entry((obj_num as u32, gen_num as u16))
            .or_default()
            .push(obj_start);
    }
    out
}

fn parse_positive_integer_indirect_object_at(
    input: &[u8],
    offset: usize,
    expected_obj_num: u32,
    expected_gen_num: u16,
) -> Option<usize> {
    let mut lexer = Lexer::new(&input[offset..]);
    match lexer.next_token()? {
        Token::Integer(v) if v == expected_obj_num as i64 => {}
        _ => return None,
    }
    match lexer.next_token()? {
        Token::Integer(v) if v == expected_gen_num as i64 => {}
        _ => return None,
    }
    match lexer.next_token()? {
        Token::Keyword(ref kw) if kw == "obj" => {}
        _ => return None,
    }
    let value = match lexer.next_token()? {
        Token::Integer(v) => positive_i64_to_usize(v)?,
        _ => return None,
    };
    match lexer.next_token()? {
        Token::Keyword(ref kw) if kw == "endobj" => Some(value),
        _ => None,
    }
}

fn parse_ascii_u64(input: &[u8], mut cursor: usize) -> Option<(u64, usize)> {
    if cursor >= input.len() || !input[cursor].is_ascii_digit() {
        return None;
    }
    let mut value = 0u64;
    while cursor < input.len() && input[cursor].is_ascii_digit() {
        value = value
            .checked_mul(10)?
            .checked_add((input[cursor] - b'0') as u64)?;
        cursor += 1;
    }
    Some((value, cursor))
}

fn positive_i64_to_usize(value: i64) -> Option<usize> {
    if value <= 0 {
        return None;
    }
    usize::try_from(value).ok()
}

fn is_pdf_whitespace(byte: u8) -> bool {
    matches!(byte, b'\x00' | b'\t' | b'\n' | b'\x0C' | b'\r' | b' ')
}

fn is_pdf_delim_or_whitespace(byte: u8) -> bool {
    is_pdf_whitespace(byte)
        || matches!(
            byte,
            b'(' | b')' | b'<' | b'>' | b'[' | b']' | b'{' | b'}' | b'/' | b'%'
        )
}

fn decode_stream_data(dict: &HashMap<String, Object>, data: &[u8]) -> Vec<u8> {
    if let Some(filter) = dict.get("Filter") {
        if let Some(name) = filter.as_name() {
            if name == "FlateDecode" {
                return flate_decode(data);
            }
        } else if let Some(arr) = filter.as_array() {
            if let Some(Object::Name(name)) = arr.get(0) {
                if name == "FlateDecode" {
                    return flate_decode(data);
                }
            }
        }
    }
    data.to_vec()
}

fn flate_decode(data: &[u8]) -> Vec<u8> {
    use std::io::Read;
    let mut decoder = flate2::read::ZlibDecoder::new(data);
    let mut out = Vec::new();
    let _ = decoder.read_to_end(&mut out);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_does_not_consume_object_number_after_startxref_offset() {
        let input = br#"%PDF-1.7
1 0 obj
<< /Type /Catalog >>
endobj
startxref
123
2 0 obj
(ok)
endobj
%%EOF
"#;
        let lexer = Lexer::new(input);
        let doc = Parser::new(lexer).parse().expect("parse");

        assert!(doc.get(1, 0).is_some(), "object 1 0 should be present");
        assert!(doc.get(2, 0).is_some(), "object 2 0 should be present");
    }

    #[test]
    fn parse_stream_uses_indirect_length_when_already_parsed() {
        let input = br#"%PDF-1.7
1 0 obj
5
endobj
2 0 obj
<< /Length 1 0 R >>
stream
ABCDE
endstream
endobj
%%EOF
"#;
        let lexer = Lexer::new(input);
        let doc = Parser::new(lexer).parse().expect("parse");
        let Some(Object::Stream { data, .. }) = doc.get(2, 0) else {
            panic!("object 2 0 should be a stream");
        };
        assert_eq!(data, b"ABCDE");
    }

    #[test]
    fn parse_stream_uses_forward_indirect_length_when_defined_later() {
        let input = br#"%PDF-1.7
2 0 obj
<< /Length 1 0 R >>
stream
ABCDE
endstream
endobj
1 0 obj
5
endobj
%%EOF
"#;
        let lexer = Lexer::new(input);
        let doc = Parser::new(lexer).parse().expect("parse");
        let Some(Object::Stream { data, .. }) = doc.get(2, 0) else {
            panic!("object 2 0 should be a stream");
        };
        assert_eq!(data, b"ABCDE");
    }

    #[test]
    fn parse_stream_falls_back_to_endstream_when_length_ref_missing() {
        let input = br#"%PDF-1.7
2 0 obj
<< /Length 9 0 R >>
stream
ABCDE
endstream
endobj
%%EOF
"#;
        let lexer = Lexer::new(input);
        let doc = Parser::new(lexer).parse().expect("parse");
        let Some(Object::Stream { data, .. }) = doc.get(2, 0) else {
            panic!("object 2 0 should be a stream");
        };
        assert_eq!(data, b"ABCDE\n");
    }

    #[test]
    fn parse_stream_recovers_when_resolved_length_is_too_short() {
        let input = br#"%PDF-1.7
2 0 obj
<< /Length 1 0 R >>
stream
ABCDE
endstream
endobj
1 0 obj
3
endobj
3 0 obj
(ok)
endobj
%%EOF
"#;
        let lexer = Lexer::new(input);
        let doc = Parser::new(lexer).parse().expect("parse");
        let Some(Object::Stream { data, .. }) = doc.get(2, 0) else {
            panic!("object 2 0 should be a stream");
        };
        assert_eq!(data, b"ABCDE\n");
        assert!(
            doc.get(3, 0).is_some(),
            "object after stream should still parse"
        );
    }

    #[test]
    fn parse_stream_does_not_scan_when_endstream_is_missing_but_endobj_follows() {
        let input = br#"%PDF-1.7
1 0 obj
3
endobj
2 0 obj
<< /Length 1 0 R >>
stream
ABC
endobj
3 0 obj
(ok)
endobj
%%EOF
"#;
        let lexer = Lexer::new(input);
        let doc = Parser::new(lexer).parse().expect("parse");
        let Some(Object::Stream { data, .. }) = doc.get(2, 0) else {
            panic!("object 2 0 should be a stream");
        };
        assert_eq!(data, b"ABC");
        assert!(
            doc.get(3, 0).is_some(),
            "object after stream should still parse"
        );
    }

    #[test]
    fn scan_indirect_object_headers_finds_expected_offsets() {
        let input = br#"%PDF-1.7
2 0 obj
<< /Length 1 0 R >>
stream
ABCDE
endstream
endobj
1 0 obj
5
endobj
%%EOF
"#;
        let index = scan_indirect_object_headers(input);
        assert!(index.contains_key(&(1, 0)));
        assert!(index.contains_key(&(2, 0)));
    }

    #[test]
    fn parse_positive_integer_indirect_object_at_reads_length_object() {
        let input = br#"%PDF-1.7
1 0 obj
5
endobj
%%EOF
"#;
        let idx = input
            .windows(7)
            .position(|w| w == b"1 0 obj")
            .expect("find object header");
        let value = parse_positive_integer_indirect_object_at(input, idx, 1, 0)
            .expect("parse integer indirect object");
        assert_eq!(value, 5);
    }

    #[test]
    fn resolve_forward_length_reference_finds_later_integer_object() {
        let input = br#"%PDF-1.7
2 0 obj
<< /Length 1 0 R >>
stream
ABCDE
endstream
endobj
1 0 obj
5
endobj
%%EOF
"#;
        let index = scan_indirect_object_headers(input);
        let offsets = index.get(&(1, 0)).expect("offsets for 1 0");
        assert!(!offsets.is_empty());
        assert!(
            offsets.iter().copied().any(|offset| {
                parse_positive_integer_indirect_object_at(input, offset, 1, 0).is_some()
            }),
            "expected at least one valid 1 0 obj candidate"
        );
        let mut parser = Parser::new(Lexer::new(input));
        assert_eq!(parser.resolve_forward_length_reference(1, 0), Some(5));
    }
}
