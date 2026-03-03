#[derive(Debug, PartialEq, Clone)]
pub enum Token {
    Null,
    Boolean(bool),
    Integer(i64),
    Real(f64),
    String(Vec<u8>),
    HexString(Vec<u8>),
    Name(String),
    Keyword(Keyword),

    DictStart,
    DictEnd,
    ArrayStart,
    ArrayEnd,
}

#[derive(Debug, PartialEq, Clone)]
pub enum Keyword {
    Obj,
    EndObj,
    Stream,
    EndStream,
    Trailer,
    Ref,
    Opq,
    OpQ,
    OpCm,
    OpW,
    OpJ,
    Opj,
    OpM,
    Opd,
    Opri,
    Opi,
    Opgs,
    Opm,
    Opl,
    Opc,
    Opv,
    Opy,
    Oph,
    Opre,
    OpS,
    Ops,
    Opf,
    OpF,
    OpfStar,
    OpB,
    OpBStar,
    Opb,
    OpbStar,
    Opn,
    OpWClip,
    OpWClipStar,
    OpCS,
    Opcs,
    OpSC,
    OpSCN,
    Opsc,
    Opscn,
    OpG,
    Opg,
    OpRG,
    Oprg,
    OpK,
    Opk,
    Opsh,
    OpBT,
    OpET,
    OpTc,
    OpTw,
    OpTz,
    OpTL,
    OpTf,
    OpTr,
    OpTs,
    OpTd,
    OpTD,
    OpTm,
    OpTStar,
    OpTj,
    OpTJ,
    OpApostrophe,
    OpQuote,
    Opd0,
    Opd1,
    OpBI,
    OpID,
    OpEI,
    OpMP,
    OpDP,
    OpBMC,
    OpBDC,
    OpEMC,
    OpBX,
    OpEX,
    OpDo,
    Other(String),
}

impl Keyword {
    pub fn as_str(&self) -> &str {
        match self {
            Keyword::Obj => "obj",
            Keyword::EndObj => "endobj",
            Keyword::Stream => "stream",
            Keyword::EndStream => "endstream",
            Keyword::Trailer => "trailer",
            Keyword::Ref => "R",
            Keyword::Opq => "q",
            Keyword::OpQ => "Q",
            Keyword::OpCm => "cm",
            Keyword::OpW => "w",
            Keyword::OpJ => "J",
            Keyword::Opj => "j",
            Keyword::OpM => "M",
            Keyword::Opd => "d",
            Keyword::Opri => "ri",
            Keyword::Opi => "i",
            Keyword::Opgs => "gs",
            Keyword::Opm => "m",
            Keyword::Opl => "l",
            Keyword::Opc => "c",
            Keyword::Opv => "v",
            Keyword::Opy => "y",
            Keyword::Oph => "h",
            Keyword::Opre => "re",
            Keyword::OpS => "S",
            Keyword::Ops => "s",
            Keyword::Opf => "f",
            Keyword::OpF => "F",
            Keyword::OpfStar => "f*",
            Keyword::OpB => "B",
            Keyword::OpBStar => "B*",
            Keyword::Opb => "b",
            Keyword::OpbStar => "b*",
            Keyword::Opn => "n",
            Keyword::OpWClip => "W",
            Keyword::OpWClipStar => "W*",
            Keyword::OpCS => "CS",
            Keyword::Opcs => "cs",
            Keyword::OpSC => "SC",
            Keyword::OpSCN => "SCN",
            Keyword::Opsc => "sc",
            Keyword::Opscn => "scn",
            Keyword::OpG => "G",
            Keyword::Opg => "g",
            Keyword::OpRG => "RG",
            Keyword::Oprg => "rg",
            Keyword::OpK => "K",
            Keyword::Opk => "k",
            Keyword::Opsh => "sh",
            Keyword::OpBT => "BT",
            Keyword::OpET => "ET",
            Keyword::OpTc => "Tc",
            Keyword::OpTw => "Tw",
            Keyword::OpTz => "Tz",
            Keyword::OpTL => "TL",
            Keyword::OpTf => "Tf",
            Keyword::OpTr => "Tr",
            Keyword::OpTs => "Ts",
            Keyword::OpTd => "Td",
            Keyword::OpTD => "TD",
            Keyword::OpTm => "Tm",
            Keyword::OpTStar => "T*",
            Keyword::OpTj => "Tj",
            Keyword::OpTJ => "TJ",
            Keyword::OpApostrophe => "'",
            Keyword::OpQuote => "\"",
            Keyword::Opd0 => "d0",
            Keyword::Opd1 => "d1",
            Keyword::OpBI => "BI",
            Keyword::OpID => "ID",
            Keyword::OpEI => "EI",
            Keyword::OpMP => "MP",
            Keyword::OpDP => "DP",
            Keyword::OpBMC => "BMC",
            Keyword::OpBDC => "BDC",
            Keyword::OpEMC => "EMC",
            Keyword::OpBX => "BX",
            Keyword::OpEX => "EX",
            Keyword::OpDo => "Do",
            Keyword::Other(v) => v.as_str(),
        }
    }

    pub fn is_content_stream_operator(&self) -> bool {
        matches!(
            self,
            Keyword::Opq
                | Keyword::OpQ
                | Keyword::OpCm
                | Keyword::OpW
                | Keyword::OpJ
                | Keyword::Opj
                | Keyword::OpM
                | Keyword::Opd
                | Keyword::Opri
                | Keyword::Opi
                | Keyword::Opgs
                | Keyword::Opm
                | Keyword::Opl
                | Keyword::Opc
                | Keyword::Opv
                | Keyword::Opy
                | Keyword::Oph
                | Keyword::Opre
                | Keyword::OpS
                | Keyword::Ops
                | Keyword::Opf
                | Keyword::OpF
                | Keyword::OpfStar
                | Keyword::OpB
                | Keyword::OpBStar
                | Keyword::Opb
                | Keyword::OpbStar
                | Keyword::Opn
                | Keyword::OpWClip
                | Keyword::OpWClipStar
                | Keyword::OpCS
                | Keyword::Opcs
                | Keyword::OpSC
                | Keyword::OpSCN
                | Keyword::Opsc
                | Keyword::Opscn
                | Keyword::OpG
                | Keyword::Opg
                | Keyword::OpRG
                | Keyword::Oprg
                | Keyword::OpK
                | Keyword::Opk
                | Keyword::Opsh
                | Keyword::OpBT
                | Keyword::OpET
                | Keyword::OpTc
                | Keyword::OpTw
                | Keyword::OpTz
                | Keyword::OpTL
                | Keyword::OpTf
                | Keyword::OpTr
                | Keyword::OpTs
                | Keyword::OpTd
                | Keyword::OpTD
                | Keyword::OpTm
                | Keyword::OpTStar
                | Keyword::OpTj
                | Keyword::OpTJ
                | Keyword::OpApostrophe
                | Keyword::OpQuote
                | Keyword::Opd0
                | Keyword::Opd1
                | Keyword::OpBI
                | Keyword::OpID
                | Keyword::OpEI
                | Keyword::OpMP
                | Keyword::OpDP
                | Keyword::OpBMC
                | Keyword::OpBDC
                | Keyword::OpEMC
                | Keyword::OpBX
                | Keyword::OpEX
                | Keyword::OpDo
        ) || matches!(self, Keyword::Other(op) if is_content_stream_operator_str(op))
    }
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
                        let (start, end) = self.read_word_bounds();
                        return Some(self.word_to_token(&self.input[start..end]));
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
                return op.is_content_stream_operator();
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
        let start = self.pos;
        while self.pos < self.input.len() {
            let byte = self.input[self.pos];
            if is_delim(byte) || is_whitespace(byte) {
                break;
            }
            self.pos += 1;
        }
        std::str::from_utf8(&self.input[start..self.pos])
            .unwrap_or_default()
            .to_string()
    }

    fn read_number(&mut self, first: u8) -> Token {
        let start = self.pos.saturating_sub(1);
        let mut has_dot = first == b'.';
        while self.pos < self.input.len() {
            let byte = self.input[self.pos];
            if is_delim(byte) || is_whitespace(byte) {
                break;
            }
            if byte == b'.' {
                has_dot = true;
            }
            self.pos += 1;
        }
        let s = std::str::from_utf8(&self.input[start..self.pos]).unwrap_or_default();
        if has_dot {
            Token::Real(s.parse::<f64>().unwrap_or(0.0))
        } else {
            Token::Integer(s.parse::<i64>().unwrap_or(0))
        }
    }

    fn read_word_bounds(&mut self) -> (usize, usize) {
        let start = self.pos.saturating_sub(1);
        while self.pos < self.input.len() {
            let byte = self.input[self.pos];
            if is_delim(byte) || is_whitespace(byte) {
                break;
            }
            self.pos += 1;
        }
        (start, self.pos)
    }

    fn word_to_token(&self, word: &[u8]) -> Token {
        match word {
            b"true" => Token::Boolean(true),
            b"false" => Token::Boolean(false),
            b"null" => Token::Null,
            _ => Token::Keyword(keyword_from_bytes(word)),
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

fn keyword_from_bytes(word: &[u8]) -> Keyword {
    match word {
        b"obj" => Keyword::Obj,
        b"endobj" => Keyword::EndObj,
        b"stream" => Keyword::Stream,
        b"endstream" => Keyword::EndStream,
        b"trailer" => Keyword::Trailer,
        b"R" => Keyword::Ref,
        b"q" => Keyword::Opq,
        b"Q" => Keyword::OpQ,
        b"cm" => Keyword::OpCm,
        b"w" => Keyword::OpW,
        b"J" => Keyword::OpJ,
        b"j" => Keyword::Opj,
        b"M" => Keyword::OpM,
        b"d" => Keyword::Opd,
        b"ri" => Keyword::Opri,
        b"i" => Keyword::Opi,
        b"gs" => Keyword::Opgs,
        b"m" => Keyword::Opm,
        b"l" => Keyword::Opl,
        b"c" => Keyword::Opc,
        b"v" => Keyword::Opv,
        b"y" => Keyword::Opy,
        b"h" => Keyword::Oph,
        b"re" => Keyword::Opre,
        b"S" => Keyword::OpS,
        b"s" => Keyword::Ops,
        b"f" => Keyword::Opf,
        b"F" => Keyword::OpF,
        b"f*" => Keyword::OpfStar,
        b"B" => Keyword::OpB,
        b"B*" => Keyword::OpBStar,
        b"b" => Keyword::Opb,
        b"b*" => Keyword::OpbStar,
        b"n" => Keyword::Opn,
        b"W" => Keyword::OpWClip,
        b"W*" => Keyword::OpWClipStar,
        b"CS" => Keyword::OpCS,
        b"cs" => Keyword::Opcs,
        b"SC" => Keyword::OpSC,
        b"SCN" => Keyword::OpSCN,
        b"sc" => Keyword::Opsc,
        b"scn" => Keyword::Opscn,
        b"G" => Keyword::OpG,
        b"g" => Keyword::Opg,
        b"RG" => Keyword::OpRG,
        b"rg" => Keyword::Oprg,
        b"K" => Keyword::OpK,
        b"k" => Keyword::Opk,
        b"sh" => Keyword::Opsh,
        b"BT" => Keyword::OpBT,
        b"ET" => Keyword::OpET,
        b"Tc" => Keyword::OpTc,
        b"Tw" => Keyword::OpTw,
        b"Tz" => Keyword::OpTz,
        b"TL" => Keyword::OpTL,
        b"Tf" => Keyword::OpTf,
        b"Tr" => Keyword::OpTr,
        b"Ts" => Keyword::OpTs,
        b"Td" => Keyword::OpTd,
        b"TD" => Keyword::OpTD,
        b"Tm" => Keyword::OpTm,
        b"T*" => Keyword::OpTStar,
        b"Tj" => Keyword::OpTj,
        b"TJ" => Keyword::OpTJ,
        b"'" => Keyword::OpApostrophe,
        b"\"" => Keyword::OpQuote,
        b"d0" => Keyword::Opd0,
        b"d1" => Keyword::Opd1,
        b"BI" => Keyword::OpBI,
        b"ID" => Keyword::OpID,
        b"EI" => Keyword::OpEI,
        b"MP" => Keyword::OpMP,
        b"DP" => Keyword::OpDP,
        b"BMC" => Keyword::OpBMC,
        b"BDC" => Keyword::OpBDC,
        b"EMC" => Keyword::OpEMC,
        b"BX" => Keyword::OpBX,
        b"EX" => Keyword::OpEX,
        b"Do" => Keyword::OpDo,
        _ => Keyword::Other(std::str::from_utf8(word).unwrap_or_default().to_string()),
    }
}

fn is_content_stream_operator_str(op: &str) -> bool {
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
        assert_eq!(lexer.next(), Some(Token::Keyword(Keyword::EndStream)));
    }

    #[test]
    fn consume_stream_until_endstream_handles_crlf() {
        let input = b"stream\r\nxyzendstream";
        let mut lexer = Lexer::new(input);
        lexer.set_position(6);

        let stream = lexer.consume_stream_until_endstream();
        assert_eq!(stream, b"xyz".to_vec());
        assert_eq!(lexer.position(), 11);
        assert_eq!(lexer.next(), Some(Token::Keyword(Keyword::EndStream)));
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
        assert_eq!(lexer.next(), Some(Token::Keyword(Keyword::OpQ)));
    }

    #[test]
    fn skip_inline_image_data_ignores_embedded_ei_with_invalid_followup_and_finds_real_terminator()
    {
        let input = b"ID \xffEI zzzz EI Q";
        let mut lexer = Lexer::new(input);
        lexer.set_position(2);

        lexer.skip_inline_image_data();
        assert_eq!(lexer.next(), Some(Token::Keyword(Keyword::OpQ)));
    }

    #[test]
    fn tokenizer_maps_common_words_to_keyword_enum() {
        let input = b"obj endobj stream endstream trailer R Tf TJ";
        let mut lexer = Lexer::new(input);
        assert_eq!(lexer.next(), Some(Token::Keyword(Keyword::Obj)));
        assert_eq!(lexer.next(), Some(Token::Keyword(Keyword::EndObj)));
        assert_eq!(lexer.next(), Some(Token::Keyword(Keyword::Stream)));
        assert_eq!(lexer.next(), Some(Token::Keyword(Keyword::EndStream)));
        assert_eq!(lexer.next(), Some(Token::Keyword(Keyword::Trailer)));
        assert_eq!(lexer.next(), Some(Token::Keyword(Keyword::Ref)));
        assert_eq!(lexer.next(), Some(Token::Keyword(Keyword::OpTf)));
        assert_eq!(lexer.next(), Some(Token::Keyword(Keyword::OpTJ)));
        assert_eq!(lexer.next(), None);
    }

    #[test]
    fn tokenizer_keeps_unknown_words_as_keyword_other() {
        let input = b"CustomOperator";
        let mut lexer = Lexer::new(input);
        assert_eq!(
            lexer.next(),
            Some(Token::Keyword(Keyword::Other("CustomOperator".to_string())))
        );
        assert_eq!(lexer.next(), None);
    }
}
