use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq)]
pub enum Object {
    Null,
    Boolean(bool),
    Integer(i64),
    Real(f64),
    String(Vec<u8>),
    Name(String),
    Array(Vec<Object>),
    Dictionary(HashMap<String, Object>),
    Stream {
        dict: HashMap<String, Object>,
        data: Vec<u8>,
    },
    Reference {
        obj_num: u32,
        gen_num: u16,
    },
}

impl Object {
    pub fn as_i64(&self) -> Option<i64> {
        match self {
            Object::Integer(v) => Some(*v),
            _ => None,
        }
    }

    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Object::Real(v) => Some(*v),
            Object::Integer(v) => Some(*v as f64),
            _ => None,
        }
    }

    pub fn as_name(&self) -> Option<&str> {
        match self {
            Object::Name(v) => Some(v),
            _ => None,
        }
    }

    pub fn as_dict(&self) -> Option<&HashMap<String, Object>> {
        match self {
            Object::Dictionary(v) => Some(v),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<&[Object]> {
        match self {
            Object::Array(v) => Some(v),
            _ => None,
        }
    }
}
