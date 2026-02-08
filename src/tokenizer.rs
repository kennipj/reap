#[derive(Debug, PartialEq, Clone)]
pub enum Token {
    Null,
    Boolean(bool),
    Integer(i64),
    Real(f64),
    String(Vec<u8>),
    HexString(Vec<u8>),
    Name(String),
    Keyword(String),

    DictStart,
    DictEnd,
    ArrayStart,
    ArrayEnd,
}

pub struct Lexer<'a> {
    input: &'a [u8],
    pos: usize,
}

impl Lexer<'_> {
    pub fn new<'a>(input: &'a [u8]) -> Lexer<'a> {
        Lexer {
            input: input,
            pos: 0,
        }
    }

    pub fn from_file(file_name: &str) -> std::io::Result<Lexer<'static>> {
        let data = std::fs::read(file_name)?;
        Ok(Lexer::new(Box::leak(data.into_boxed_slice())))
    }

    pub fn from_bytes(data: Vec<u8>) -> Lexer<'static> {
        Lexer::new(Box::leak(data.into_boxed_slice()))
    }
}

impl Iterator for Lexer<'_> {
    type Item = Token;
    fn next(&mut self) -> Option<Self::Item> {
        self.next_token()
    }
}

impl Lexer<'_> {
    pub fn position(&self) -> usize {
        self.pos
    }

    pub(crate) fn input(&self) -> &[u8] {
        self.input
    }

    pub fn set_position(&mut self, pos: usize) {
        self.pos = pos;
    }

    pub fn next_token(&mut self) -> Option<Token> {
        while self.pos < self.input.len() {
            self.skip_whitespace_and_comments();
            if self.pos >= self.input.len() {
                return None;
            }

            let byte = self.input[self.pos];
            self.pos += 1;
            match byte {
                b'[' => return Some(Token::ArrayStart),
                b']' => return Some(Token::ArrayEnd),
                b'<' if self.pos < self.input.len() && self.input[self.pos] == b'<' => {
                    self.pos += 1;
                    return Some(Token::DictStart);
                }
                b'>' if self.pos < self.input.len() && self.input[self.pos] == b'>' => {
                    self.pos += 1;
                    return Some(Token::DictEnd);
                }
                b'(' => return Some(Token::String(self.read_literal_string())),
                b'<' => return Some(Token::HexString(self.read_hex_string())),
                b'/' => return Some(Token::Name(self.read_name())),
                b'+' | b'-' | b'.' | b'0'..=b'9' => return Some(self.read_number(byte)),
                _ => {
                    if is_regular(byte) {
                        let word = self.read_word(byte);
                        return Some(self.word_to_token(word));
                    }
                }
            }
        }
        None
    }

    pub fn consume_stream(&mut self, length: usize) -> Vec<u8> {
        self.skip_stream_linebreak();
        let start = self.pos;
        let end = (start + length).min(self.input.len());
        self.pos = end;
        self.input[start..end].to_vec()
    }

    pub fn consume_stream_until_endstream(&mut self) -> Vec<u8> {
        self.skip_stream_linebreak();
        let start = self.pos;
        if let Some(found) = memchr::memmem::find(&self.input[start..], b"endstream") {
            let endstream_pos = start + found;
            let data = self.input[start..endstream_pos].to_vec();
            self.pos = endstream_pos;
            return data;
        }
        let data = self.input[start..].to_vec();
        self.pos = self.input.len();
        data
    }

    pub fn skip_inline_image_data(&mut self) {
        if self.pos >= self.input.len() {
            return;
        }

        // The ID operator is followed by a single required whitespace byte.
        if self.input[self.pos] == b'\r' {
            self.pos += 1;
            if self.pos < self.input.len() && self.input[self.pos] == b'\n' {
                self.pos += 1;
            }
        } else if is_inline_image_whitespace(self.input[self.pos]) {
            self.pos += 1;
        }

        let mut i = self.pos;
        while i + 1 < self.input.len() {
            if self.input[i] == b'E' && self.input[i + 1] == b'I' {
                let prev_ok = i > 0 && is_inline_image_whitespace(self.input[i - 1]);
                let next_ok = i + 2 >= self.input.len()
                    || is_inline_image_whitespace(self.input[i + 2])
                    || is_delim(self.input[i + 2]);
                if prev_ok && next_ok {
                    self.pos = i + 2;
                    return;
                }
            }
            i += 1;
        }

        let mut i = self.pos;
        while i + 1 < self.input.len() {
            if self.input[i] == b'E' && self.input[i + 1] == b'I' {
                let next_ok = i + 2 >= self.input.len()
                    || is_inline_image_whitespace(self.input[i + 2])
                    || is_delim(self.input[i + 2]);
                if next_ok && self.looks_like_content_stream_followup(i + 2) {
                    self.pos = i + 2;
                    return;
                }
            }
            i += 1;
        }

        self.pos = self.input.len();
    }

    fn looks_like_content_stream_followup(&self, after_ei_pos: usize) -> bool {
        let mut lookahead_pos = after_ei_pos;
        while lookahead_pos < self.input.len() && is_whitespace(self.input[lookahead_pos]) {
            lookahead_pos += 1;
        }
        if lookahead_pos >= self.input.len() {
            return true;
        }

        let mut lookahead = Lexer::new(&self.input[lookahead_pos..]);
        let mut seen = 0usize;
        while seen < 8 {
            let Some(tok) = lookahead.next_token() else {
                break;
            };
            seen += 1;
            if let Token::Keyword(op) = tok {
                return is_content_stream_operator(op.as_str());
            }
        }
        false
    }

    fn skip_stream_linebreak(&mut self) {
        if self.pos < self.input.len() && self.input[self.pos] == b'\r' {
            self.pos += 1;
            if self.pos < self.input.len() && self.input[self.pos] == b'\n' {
                self.pos += 1;
            }
        } else if self.pos < self.input.len() && self.input[self.pos] == b'\n' {
            self.pos += 1;
        }
    }

    fn skip_whitespace_and_comments(&mut self) {
        loop {
            while self.pos < self.input.len() && is_whitespace(self.input[self.pos]) {
                self.pos += 1;
            }
            if self.pos < self.input.len() && self.input[self.pos] == b'%' {
                while self.pos < self.input.len() && self.input[self.pos] != b'\n' {
                    self.pos += 1;
                }
                continue;
            }
            break;
        }
    }

    fn read_literal_string(&mut self) -> Vec<u8> {
        let mut out = Vec::new();
        let mut depth = 1;
        while self.pos < self.input.len() {
            let byte = self.input[self.pos];
            self.pos += 1;
            match byte {
                b'\\' => {
                    if self.pos >= self.input.len() {
                        break;
                    }
                    let next = self.input[self.pos];
                    self.pos += 1;
                    match next {
                        b'n' => out.push(b'\n'),
                        b'r' => out.push(b'\r'),
                        b't' => out.push(b'\t'),
                        b'b' => out.push(0x08),
                        b'f' => out.push(0x0C),
                        b'\\' => out.push(b'\\'),
                        b'(' => out.push(b'('),
                        b')' => out.push(b')'),
                        b'\r' => {
                            if self.pos < self.input.len() && self.input[self.pos] == b'\n' {
                                self.pos += 1;
                            }
                        }
                        b'\n' => {}
                        b'0'..=b'7' => {
                            let mut val = (next - b'0') as u16;
                            for _ in 0..2 {
                                if self.pos >= self.input.len() {
                                    break;
                                }
                                let b = self.input[self.pos];
                                if !(b'0'..=b'7').contains(&b) {
                                    break;
                                }
                                self.pos += 1;
                                val = (val << 3) | (b - b'0') as u16;
                            }
                            out.push((val & 0xFF) as u8);
                        }
                        other => out.push(other),
                    }
                }
                b'(' => {
                    depth += 1;
                    out.push(byte);
                }
                b')' => {
                    depth -= 1;
                    if depth == 0 {
                        break;
                    }
                    out.push(byte);
                }
                _ => out.push(byte),
            }
        }
        out
    }

    fn read_hex_string(&mut self) -> Vec<u8> {
        let mut out = Vec::new();
        let mut pending_nibble: Option<u8> = None;
        while self.pos < self.input.len() {
            let byte = self.input[self.pos];
            self.pos += 1;
            if byte == b'>' {
                break;
            }
            if is_whitespace(byte) {
                continue;
            }
            if let Some(nibble) = hex_nibble(byte) {
                if let Some(high) = pending_nibble.take() {
                    out.push((high << 4) | nibble);
                } else {
                    pending_nibble = Some(nibble);
                }
            }
        }
        if let Some(high) = pending_nibble {
            out.push(high << 4);
        }
        out
    }

    fn read_name(&mut self) -> String {
        let mut out = Vec::new();
        while self.pos < self.input.len() {
            let byte = self.input[self.pos];
            if is_delim(byte) || is_whitespace(byte) {
                break;
            }
            out.push(byte);
            self.pos += 1;
        }
        String::from_utf8(out).unwrap_or_default()
    }

    fn read_number(&mut self, first: u8) -> Token {
        let mut out = Vec::new();
        out.push(first);
        while self.pos < self.input.len() {
            let byte = self.input[self.pos];
            if is_delim(byte) || is_whitespace(byte) {
                break;
            }
            out.push(byte);
            self.pos += 1;
        }
        let s = String::from_utf8(out).unwrap_or_default();
        if s.contains('.') {
            Token::Real(s.parse().unwrap_or(0.0))
        } else {
            Token::Integer(s.parse().unwrap_or(0))
        }
    }

    fn read_word(&mut self, first: u8) -> String {
        let mut out = Vec::new();
        out.push(first);
        while self.pos < self.input.len() {
            let byte = self.input[self.pos];
            if is_delim(byte) || is_whitespace(byte) {
                break;
            }
            out.push(byte);
            self.pos += 1;
        }
        String::from_utf8(out).unwrap_or_default()
    }

    fn word_to_token(&self, word: String) -> Token {
        match word.as_str() {
            "true" => Token::Boolean(true),
            "false" => Token::Boolean(false),
            "null" => Token::Null,
            _ => Token::Keyword(word),
        }
    }
}

fn is_whitespace(byte: u8) -> bool {
    matches!(byte, b'\x00' | b'\x09' | b'\x0a' | b'\x0c' | b'\x0d' | b' ')
}

fn is_inline_image_whitespace(byte: u8) -> bool {
    matches!(byte, b'\x09' | b'\x0a' | b'\x0c' | b'\x0d' | b' ')
}

fn is_delim(byte: u8) -> bool {
    matches!(
        byte,
        b'(' | b')' | b'<' | b'>' | b'[' | b']' | b'{' | b'}' | b'/' | b'%'
    )
}

fn is_regular(byte: u8) -> bool {
    !(is_delim(byte) || is_whitespace(byte))
}

fn is_content_stream_operator(op: &str) -> bool {
    matches!(
        op,
        "q" | "Q"
            | "cm"
            | "w"
            | "J"
            | "j"
            | "M"
            | "d"
            | "ri"
            | "i"
            | "gs"
            | "m"
            | "l"
            | "c"
            | "v"
            | "y"
            | "h"
            | "re"
            | "S"
            | "s"
            | "f"
            | "F"
            | "f*"
            | "B"
            | "B*"
            | "b"
            | "b*"
            | "n"
            | "W"
            | "W*"
            | "CS"
            | "cs"
            | "SC"
            | "SCN"
            | "sc"
            | "scn"
            | "G"
            | "g"
            | "RG"
            | "rg"
            | "K"
            | "k"
            | "sh"
            | "BT"
            | "ET"
            | "Tc"
            | "Tw"
            | "Tz"
            | "TL"
            | "Tf"
            | "Tr"
            | "Ts"
            | "Td"
            | "TD"
            | "Tm"
            | "T*"
            | "Tj"
            | "TJ"
            | "'"
            | "\""
            | "d0"
            | "d1"
            | "BI"
            | "ID"
            | "EI"
            | "MP"
            | "DP"
            | "BMC"
            | "BDC"
            | "EMC"
            | "BX"
            | "EX"
            | "Do"
    )
}

fn hex_nibble(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        b'A'..=b'F' => Some(byte - b'A' + 10),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_array_of_integers() {
        let input = b"[1 2 3]";
        let mut lexer = Lexer::new(input);
        assert_eq!(lexer.next(), Some(Token::ArrayStart));
        assert_eq!(lexer.next(), Some(Token::Integer(1)));
        assert_eq!(lexer.next(), Some(Token::Integer(2)));
        assert_eq!(lexer.next(), Some(Token::Integer(3)));
        assert_eq!(lexer.next(), Some(Token::ArrayEnd));
        assert_eq!(lexer.next(), None);
    }

    #[test]
    fn test_dictionary() {
        let input = b"<< /Type /Example >>";
        let mut lexer = Lexer::new(input);
        assert_eq!(lexer.next(), Some(Token::DictStart));
        assert_eq!(lexer.next(), Some(Token::Name("Type".to_string())));
        assert_eq!(lexer.next(), Some(Token::Name("Example".to_string())));
        assert_eq!(lexer.next(), Some(Token::DictEnd));
        assert_eq!(lexer.next(), None);
    }

    #[test]
    fn test_string() {
        let input = b"(Hello, World!)";
        let mut lexer = Lexer::new(input);
        assert_eq!(lexer.next(), Some(Token::String(b"Hello, World!".to_vec())));
        assert_eq!(lexer.next(), None);
    }

    #[test]
    fn test_hex_string_decodes_bytes() {
        let input = b"<656592>";
        let mut lexer = Lexer::new(input);
        assert_eq!(lexer.next(), Some(Token::HexString(vec![0x65, 0x65, 0x92])));
        assert_eq!(lexer.next(), None);
    }

    #[test]
    fn test_hex_string_odd_nibbles_pad_low_nibble() {
        let input = b"<4E6F7>";
        let mut lexer = Lexer::new(input);
        assert_eq!(lexer.next(), Some(Token::HexString(vec![0x4E, 0x6F, 0x70])));
        assert_eq!(lexer.next(), None);
    }

    #[test]
    fn consume_stream_until_endstream_handles_lf() {
        let input = b"stream\nabcendstream\n";
        let mut lexer = Lexer::new(input);
        lexer.set_position(6);

        let stream = lexer.consume_stream_until_endstream();
        assert_eq!(stream, b"abc".to_vec());
        assert_eq!(lexer.position(), 10);
        assert_eq!(lexer.next(), Some(Token::Keyword("endstream".to_string())));
    }

    #[test]
    fn consume_stream_until_endstream_handles_crlf() {
        let input = b"stream\r\nxyzendstream";
        let mut lexer = Lexer::new(input);
        lexer.set_position(6);

        let stream = lexer.consume_stream_until_endstream();
        assert_eq!(stream, b"xyz".to_vec());
        assert_eq!(lexer.position(), 11);
        assert_eq!(lexer.next(), Some(Token::Keyword("endstream".to_string())));
    }

    #[test]
    fn consume_stream_until_endstream_excludes_endstream_marker() {
        let input = b"stream\npayloadendstreamtrailer";
        let mut lexer = Lexer::new(input);
        lexer.set_position(6);

        let stream = lexer.consume_stream_until_endstream();
        assert_eq!(stream, b"payload".to_vec());
        assert_eq!(lexer.position(), 14);
    }

    #[test]
    fn consume_stream_until_endstream_without_marker_returns_rest() {
        let input = b"stream\nrest-of-stream";
        let mut lexer = Lexer::new(input);
        lexer.set_position(6);

        let stream = lexer.consume_stream_until_endstream();
        assert_eq!(stream, b"rest-of-stream".to_vec());
        assert_eq!(lexer.position(), input.len());
    }

    #[test]
    fn skip_inline_image_data_accepts_ei_without_preceding_whitespace_when_followup_is_valid() {
        let input = b"ID \xffEI Q";
        let mut lexer = Lexer::new(input);
        lexer.set_position(2);

        lexer.skip_inline_image_data();
        assert_eq!(lexer.position(), 6);
        assert_eq!(lexer.next(), Some(Token::Keyword("Q".to_string())));
    }

    #[test]
    fn skip_inline_image_data_ignores_embedded_ei_with_invalid_followup_and_finds_real_terminator()
    {
        let input = b"ID \xffEI zzzz EI Q";
        let mut lexer = Lexer::new(input);
        lexer.set_position(2);

        lexer.skip_inline_image_data();
        assert_eq!(lexer.next(), Some(Token::Keyword("Q".to_string())));
    }
}
