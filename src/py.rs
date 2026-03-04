#![allow(unsafe_op_in_unsafe_fn)]

use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::{Arc, OnceLock};

use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBytes, PyList, PyModule, PyType};

use crate::parser::{ParseError, Parser};
use crate::rtree::{RegexSearchError, ScopedMergeRules, TextBlockIndex as RsTextBlockIndex};
use crate::text::{
    CharBBox, ExpandDirection, ExpandError, ExtractParallelMode, Rectangle as RsRectangle,
    extract_text_blocks_with_chars_and_page_rects_with_parallel_mode,
};
use crate::tokenizer::Lexer;

#[pyclass]
#[derive(Debug, Clone, PartialEq)]
struct Rectangle {
    #[pyo3(get, set)]
    top: f64,
    #[pyo3(get, set)]
    left: f64,
    #[pyo3(get, set)]
    bottom: f64,
    #[pyo3(get, set)]
    right: f64,
}

impl Rectangle {
    fn as_core(&self) -> RsRectangle {
        RsRectangle {
            top: self.top,
            left: self.left,
            bottom: self.bottom,
            right: self.right,
        }
    }

    fn with_overrides(
        &self,
        top: Option<f64>,
        left: Option<f64>,
        bottom: Option<f64>,
        right: Option<f64>,
    ) -> Self {
        Self {
            top: top.unwrap_or(self.top),
            left: left.unwrap_or(self.left),
            bottom: bottom.unwrap_or(self.bottom),
            right: right.unwrap_or(self.right),
        }
    }
}

#[pyclass]
#[derive(Clone, PartialEq)]
struct Point {
    #[pyo3(get)]
    x: f64,
    #[pyo3(get)]
    y: f64,
}

#[pymethods]
impl Point {
    fn __repr__(&self) -> String {
        format!("Point(x={}, y={})", self.x, self.y)
    }

    fn distance(&self, other: &Point) -> f64 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        (dx * dx + dy * dy).sqrt()
    }
}

#[pyclass]
#[derive(Clone)]
struct TextBlock {
    #[pyo3(get)]
    rect: Rectangle,
    #[pyo3(get)]
    text: String,
    chars_backing: Option<Arc<Vec<Vec<CharBBox>>>>,
    chars_source: BlockCharSource,
    chars_cache: OnceLock<Vec<CharBBox>>,
}

#[pyclass]
#[derive(Clone, PartialEq)]
struct TextChar {
    #[pyo3(get)]
    rect: Rectangle,
    #[pyo3(get)]
    ch: String,
}

#[derive(Clone)]
enum BlockCharSource {
    None,
    Single(usize),
    Multi(Vec<usize>),
}

#[pymethods]
impl Rectangle {
    #[new]
    fn new(top: f64, left: f64, bottom: f64, right: f64) -> Self {
        Self {
            top,
            left,
            bottom,
            right,
        }
    }

    #[pyo3(signature = (*, top=None, left=None, bottom=None, right=None))]
    fn with_coords(
        &self,
        top: Option<f64>,
        left: Option<f64>,
        bottom: Option<f64>,
        right: Option<f64>,
    ) -> Self {
        self.with_overrides(top, left, bottom, right)
    }

    fn with_top(&self, top: f64) -> Self {
        self.with_overrides(Some(top), None, None, None)
    }

    fn with_left(&self, left: f64) -> Self {
        self.with_overrides(None, Some(left), None, None)
    }

    fn with_bottom(&self, bottom: f64) -> Self {
        self.with_overrides(None, None, Some(bottom), None)
    }

    fn with_right(&self, right: f64) -> Self {
        self.with_overrides(None, None, None, Some(right))
    }

    #[pyo3(signature = (*, top=0.0, bottom=0.0, left=0.0, right=0.0))]
    fn with_margin(&self, top: f64, bottom: f64, left: f64, right: f64) -> Self {
        let normalized_left = self.left.min(self.right);
        let normalized_right = self.left.max(self.right);
        let normalized_top = self.top.min(self.bottom);
        let normalized_bottom = self.top.max(self.bottom);
        Self {
            top: normalized_top - top,
            left: normalized_left - left,
            bottom: normalized_bottom + bottom,
            right: normalized_right + right,
        }
    }

    fn expand(
        &self,
        rectangles: Vec<Rectangle>,
        directions: Vec<String>,
        maximum_bounds: Rectangle,
    ) -> PyResult<Self> {
        let mut parsed_directions = Vec::new();
        for direction in directions {
            let parsed = match direction.as_str() {
                "up" => ExpandDirection::Up,
                "down" => ExpandDirection::Down,
                "left" => ExpandDirection::Left,
                "right" => ExpandDirection::Right,
                _ => {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "invalid direction '{}'; expected one of: up, down, left, right",
                        direction
                    )));
                }
            };
            if !parsed_directions.contains(&parsed) {
                parsed_directions.push(parsed);
            }
        }
        if parsed_directions.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "directions must include at least one of: up, down, left, right",
            ));
        }

        let blockers: Vec<RsRectangle> = rectangles.iter().map(|rect| rect.as_core()).collect();
        let expanded = self
            .as_core()
            .expand_constrained(&blockers, &parsed_directions, &maximum_bounds.as_core())
            .map_err(expand_error_to_py)?;
        Ok(Self {
            top: expanded.top,
            left: expanded.left,
            bottom: expanded.bottom,
            right: expanded.right,
        })
    }

    fn get_center(&self) -> (f64, f64) {
        self.as_core().get_center()
    }

    fn overlap_percentage(&self, other: &Rectangle) -> f64 {
        self.as_core().overlap_percentage(&other.as_core())
    }

    fn overlaps_horizontally(&self, other: &Rectangle) -> bool {
        self.as_core().overlaps_horizontally(&other.as_core())
    }

    fn contains_left_side(&self, other: &Rectangle) -> bool {
        self.as_core().contains_left_side(&other.as_core())
    }

    #[pyo3(signature = (other, pct=0.0))]
    fn overlaps(&self, other: &Rectangle, pct: f64) -> bool {
        self.as_core().overlaps(&other.as_core(), pct)
    }

    fn validate(&self) -> bool {
        self.as_core().validate()
    }

    #[pyo3(signature = (other, top=0.0, bottom=0.0, left=0.0, right=0.0))]
    fn overlaps_with_margins(
        &self,
        other: &Rectangle,
        top: f64,
        bottom: f64,
        left: f64,
        right: f64,
    ) -> bool {
        self.as_core()
            .overlaps_with_margins(&other.as_core(), top, bottom, left, right)
    }

    fn vertical_overlap(&self, other: &Rectangle, threshold: f64) -> bool {
        self.as_core().vertical_overlap(&other.as_core(), threshold)
    }

    fn distance(&self, other: &Rectangle) -> f64 {
        self.as_core().distance(&other.as_core())
    }

    fn corner(&self, left_right: &str, top_bottom: &str) -> PyResult<Point> {
        let lr = match left_right {
            "left" => crate::text::LeftRight::Left,
            "right" => crate::text::LeftRight::Right,
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "left_right must be 'left' or 'right'",
                ));
            }
        };
        let tb = match top_bottom {
            "top" => crate::text::TopBottom::Top,
            "bottom" => crate::text::TopBottom::Bottom,
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "top_bottom must be 'top' or 'bottom'",
                ));
            }
        };
        let p = self.as_core().corner(lr, tb);
        Ok(Point { x: p.x, y: p.y })
    }

    fn __repr__(&self) -> String {
        format!(
            "Rectangle(top={}, left={}, bottom={}, right={})",
            self.top, self.left, self.bottom, self.right
        )
    }

    fn __eq__(&self, other: &Rectangle) -> bool {
        self.top == other.top
            && self.left == other.left
            && self.bottom == other.bottom
            && self.right == other.right
    }
}

#[pyclass(unsendable)]
struct TextBlockIndex {
    inner: RefCell<RsTextBlockIndex>,
    py_blocks: RefCell<Vec<Option<Py<TextBlock>>>>,
    regex_py_cache: RefCell<HashMap<String, Vec<Py<TextBlock>>>>,
}

#[pymethods]
impl TextBlockIndex {
    #[new]
    #[pyo3(signature = (obj, password=None, concurrency="on"))]
    fn new(
        py: Python<'_>,
        obj: &Bound<'_, PyAny>,
        password: Option<&Bound<'_, PyAny>>,
        concurrency: &str,
    ) -> PyResult<Self> {
        let password = extract_password_bytes(password)?;
        let parallel_mode = map_concurrency_to_parallel_mode(concurrency)?;
        if let Ok(py_bytes) = obj.cast::<PyBytes>() {
            return Self::from_bytes(py, py_bytes.as_bytes(), password.as_deref(), parallel_mode);
        }
        if let Ok(bytes) = obj.extract::<&[u8]>() {
            return Self::from_bytes(py, bytes, password.as_deref(), parallel_mode);
        }
        if let Ok(bytes) = obj.extract::<Vec<u8>>() {
            return Self::from_bytes(py, &bytes, password.as_deref(), parallel_mode);
        }

        let path = if obj.hasattr("__fspath__")? {
            let fs = obj.call_method0("__fspath__")?;
            fs.extract::<String>().map_err(|_| {
                pyo3::exceptions::PyTypeError::new_err(
                    "TextBlockIndex expects __fspath__ to return str path",
                )
            })?
        } else {
            obj.extract::<String>().map_err(|_| {
                pyo3::exceptions::PyTypeError::new_err(
                    "TextBlockIndex expects bytes-like object or str path",
                )
            })?
        };
        let lexer = Lexer::from_file(&path).map_err(|e| {
            pyo3::exceptions::PyIOError::new_err(format!("failed to read {}: {}", path, e))
        })?;
        Self::from_lexer(py, lexer, password.as_deref(), parallel_mode)
    }

    #[classmethod]
    #[pyo3(signature = (path, password=None, concurrency="on"))]
    fn from_path(
        _cls: &Bound<'_, PyType>,
        py: Python<'_>,
        path: &str,
        password: Option<&Bound<'_, PyAny>>,
        concurrency: &str,
    ) -> PyResult<Self> {
        let password = extract_password_bytes(password)?;
        let parallel_mode = map_concurrency_to_parallel_mode(concurrency)?;
        let lexer = Lexer::from_file(path).map_err(|e| {
            pyo3::exceptions::PyIOError::new_err(format!("failed to read {}: {}", path, e))
        })?;
        Self::from_lexer(py, lexer, password.as_deref(), parallel_mode)
    }

    fn search(&self, py: Python<'_>, rect: &Rectangle, overlap: f64) -> PyResult<Py<PyAny>> {
        let indices = self
            .inner
            .borrow_mut()
            .search_indices(rect.as_core(), overlap);
        let out = PyList::empty(py);
        for index in indices {
            if let Some(block) = self.py_block_at(py, index)? {
                out.append(block)?;
            }
        }
        Ok(out.into_any().unbind())
    }

    fn search_regex(&self, py: Python<'_>, pattern: &str) -> PyResult<Py<PyAny>> {
        if let Some(cached) = self.regex_py_cache.borrow().get(pattern) {
            let out = PyList::empty(py);
            for block in cached {
                out.append(block.clone_ref(py))?;
            }
            return Ok(out.into_any().unbind());
        }

        let matches = {
            let mut inner = self.inner.borrow_mut();
            inner
                .search_regex_indices(pattern)
                .map_err(|err| match err {
                    RegexSearchError::InvalidPattern(e) => pyo3::exceptions::PyValueError::new_err(
                        format!("invalid regex pattern: {}", e),
                    ),
                })?
        };

        let mut py_matches: Vec<Py<TextBlock>> = Vec::with_capacity(matches.len());
        let inner = self.inner.borrow();
        let shared_block_chars = inner.block_chars_arc();
        for block_indices in matches {
            if block_indices.len() == 1 {
                if let Some(existing) = self.py_block_at(py, block_indices[0])? {
                    py_matches.push(existing);
                }
                continue;
            }

            let mut merged: Option<crate::text::TextBlock> = None;
            for index in block_indices.iter().copied() {
                let Some(block) = inner.block_at(index) else {
                    continue;
                };
                merged = Some(match merged {
                    Some(mut acc) => {
                        acc.text.push(' ');
                        acc.text.push_str(&block.text);
                        acc.bbox.left = acc.bbox.left.min(block.bbox.left);
                        acc.bbox.top = acc.bbox.top.min(block.bbox.top);
                        acc.bbox.right = acc.bbox.right.max(block.bbox.right);
                        acc.bbox.bottom = acc.bbox.bottom.max(block.bbox.bottom);
                        acc
                    }
                    None => block.clone(),
                });
            }
            if let Some(block) = merged {
                let chars_source = if shared_block_chars.is_some() {
                    BlockCharSource::Multi(block_indices)
                } else {
                    BlockCharSource::None
                };
                py_matches.push(block_to_py_object(
                    py,
                    &block,
                    shared_block_chars.clone(),
                    chars_source,
                )?);
            }
        }

        let out = PyList::empty(py);
        for block in &py_matches {
            out.append(block.clone_ref(py))?;
        }
        self.regex_py_cache
            .borrow_mut()
            .insert(pattern.to_string(), py_matches);
        Ok(out.into_any().unbind())
    }

    fn has_regex(&self, pattern: &str) -> PyResult<bool> {
        let mut inner = self.inner.borrow_mut();
        inner.search_regex_exists(pattern).map_err(|err| match err {
            RegexSearchError::InvalidPattern(e) => {
                pyo3::exceptions::PyValueError::new_err(format!("invalid regex pattern: {}", e))
            }
        })
    }

    #[pyo3(signature = (
        rect,
        overlap=0.0,
        merge_threshold=0.0,
        normalize=false,
        no_numeric_pair_merge=false,
        no_date_pair_merge=false
    ))]
    fn scoped(
        &self,
        _py: Python<'_>,
        rect: &Rectangle,
        overlap: f64,
        merge_threshold: f64,
        normalize: bool,
        no_numeric_pair_merge: bool,
        no_date_pair_merge: bool,
    ) -> PyResult<Self> {
        let core_rect = rect.as_core();
        let rules = ScopedMergeRules {
            no_numeric_pair_merge,
            no_date_pair_merge,
        };
        let scoped_inner =
            self.inner
                .borrow()
                .scoped(core_rect, overlap, merge_threshold, normalize, rules);
        let py_blocks = (0..scoped_inner.block_len()).map(|_| None).collect();
        Ok(Self {
            inner: RefCell::new(scoped_inner),
            py_blocks: RefCell::new(py_blocks),
            regex_py_cache: RefCell::new(HashMap::new()),
        })
    }

    #[getter]
    fn text(&self) -> PyResult<String> {
        Ok(self.inner.borrow().text())
    }

    #[getter]
    fn text_blocks(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let out = PyList::empty(py);
        let block_count = self.py_blocks.borrow().len();
        for index in 0..block_count {
            if let Some(block) = self.py_block_at(py, index)? {
                out.append(block)?;
            }
        }
        Ok(out.into_any().unbind())
    }

    #[getter]
    fn doc_rect(&self) -> Rectangle {
        let rect = self.inner.borrow().doc_rect();
        Rectangle {
            top: rect.top,
            left: rect.left,
            bottom: rect.bottom,
            right: rect.right,
        }
    }

    #[getter]
    fn page_rects(&self) -> Vec<Rectangle> {
        let inner = self.inner.borrow();
        inner
            .page_rects_slice()
            .iter()
            .map(|rect| Rectangle {
                top: rect.top,
                left: rect.left,
                bottom: rect.bottom,
                right: rect.right,
            })
            .collect()
    }
}

impl TextBlockIndex {
    fn from_bytes(
        _py: Python<'_>,
        bytes: &[u8],
        password: Option<&[u8]>,
        parallel_mode: ExtractParallelMode,
    ) -> PyResult<Self> {
        let doc = Parser::new(Lexer::new(bytes))
            .parse_with_password(password)
            .map_err(parse_error_to_py)?;
        let (blocks, block_chars, page_rects) =
            extract_text_blocks_with_chars_and_page_rects_with_parallel_mode(
                &doc,
                Some(parallel_mode),
            );
        let py_blocks = (0..blocks.len()).map(|_| None).collect();
        Ok(Self {
            inner: RefCell::new(RsTextBlockIndex::new_with_chars_and_page_rects(
                blocks,
                block_chars,
                page_rects,
            )),
            py_blocks: RefCell::new(py_blocks),
            regex_py_cache: RefCell::new(HashMap::new()),
        })
    }

    fn from_lexer(
        _py: Python<'_>,
        lexer: Lexer,
        password: Option<&[u8]>,
        parallel_mode: ExtractParallelMode,
    ) -> PyResult<Self> {
        let doc = Parser::new(lexer)
            .parse_with_password(password)
            .map_err(parse_error_to_py)?;
        let (blocks, block_chars, page_rects) =
            extract_text_blocks_with_chars_and_page_rects_with_parallel_mode(
                &doc,
                Some(parallel_mode),
            );
        let py_blocks = (0..blocks.len()).map(|_| None).collect();
        Ok(Self {
            inner: RefCell::new(RsTextBlockIndex::new_with_chars_and_page_rects(
                blocks,
                block_chars,
                page_rects,
            )),
            py_blocks: RefCell::new(py_blocks),
            regex_py_cache: RefCell::new(HashMap::new()),
        })
    }

    fn py_block_at(&self, py: Python<'_>, index: usize) -> PyResult<Option<Py<TextBlock>>> {
        {
            let py_blocks = self.py_blocks.borrow();
            let Some(slot) = py_blocks.get(index) else {
                return Ok(None);
            };
            if let Some(block) = slot {
                return Ok(Some(block.clone_ref(py)));
            }
        }

        let block = {
            let inner = self.inner.borrow();
            let block = inner.block_at(index).cloned();
            let chars_backing = inner.block_chars_arc();
            (block, chars_backing)
        };
        let (block, chars_backing) = block;
        let Some(block) = block else {
            return Ok(None);
        };
        let chars_source = if chars_backing.is_some() {
            BlockCharSource::Single(index)
        } else {
            BlockCharSource::None
        };
        let py_block = block_to_py_object(py, &block, chars_backing, chars_source)?;

        let mut py_blocks = self.py_blocks.borrow_mut();
        let Some(slot) = py_blocks.get_mut(index) else {
            return Ok(None);
        };
        if slot.is_none() {
            *slot = Some(py_block.clone_ref(py));
            return Ok(Some(py_block));
        }
        Ok(slot.as_ref().map(|block| block.clone_ref(py)))
    }
}

fn map_concurrency_to_parallel_mode(concurrency: &str) -> PyResult<ExtractParallelMode> {
    match concurrency.trim().to_ascii_lowercase().as_str() {
        "on" => Ok(ExtractParallelMode::On),
        "off" => Ok(ExtractParallelMode::Off),
        "auto" => Ok(ExtractParallelMode::Auto),
        _ => Err(pyo3::exceptions::PyValueError::new_err(
            "concurrency must be one of: 'on', 'off', 'auto'",
        )),
    }
}

fn extract_password_bytes(password: Option<&Bound<'_, PyAny>>) -> PyResult<Option<Vec<u8>>> {
    let Some(password) = password else {
        return Ok(None);
    };
    if password.is_none() {
        return Ok(None);
    }
    if let Ok(text) = password.extract::<String>() {
        return Ok(Some(text.into_bytes()));
    }
    if let Ok(bytes) = password.extract::<Vec<u8>>() {
        return Ok(Some(bytes));
    }
    Err(pyo3::exceptions::PyTypeError::new_err(
        "password must be str, bytes-like, or None",
    ))
}

fn parse_error_to_py(err: ParseError) -> PyErr {
    match err {
        ParseError::InvalidPassword => {
            pyo3::exceptions::PyValueError::new_err("invalid password for encrypted PDF")
        }
        ParseError::UnsupportedEncryption(msg) => pyo3::exceptions::PyRuntimeError::new_err(
            format!("unsupported PDF encryption: {}", msg),
        ),
        ParseError::MalformedEncryption(msg) => {
            pyo3::exceptions::PyValueError::new_err(format!("malformed encrypted PDF: {}", msg))
        }
    }
}

fn expand_error_to_py(err: ExpandError) -> PyErr {
    match err {
        ExpandError::EmptyDirections => pyo3::exceptions::PyValueError::new_err(
            "directions must include at least one of: up, down, left, right",
        ),
        ExpandError::InvalidMaximumBounds => pyo3::exceptions::PyValueError::new_err(
            "maximum_bounds must fully contain the rectangle being expanded",
        ),
    }
}

fn char_to_py_object(py: Python<'_>, ch: &CharBBox) -> PyResult<Py<TextChar>> {
    let rect = Rectangle {
        top: ch.bbox.top,
        left: ch.bbox.left,
        bottom: ch.bbox.bottom,
        right: ch.bbox.right,
    };
    Py::new(
        py,
        TextChar {
            rect,
            ch: ch.ch.to_string(),
        },
    )
}

fn block_to_py_object(
    py: Python<'_>,
    b: &crate::text::TextBlock,
    chars_backing: Option<Arc<Vec<Vec<CharBBox>>>>,
    chars_source: BlockCharSource,
) -> PyResult<Py<TextBlock>> {
    let rect = Rectangle {
        top: b.bbox.top,
        left: b.bbox.left,
        bottom: b.bbox.bottom,
        right: b.bbox.right,
    };
    Py::new(
        py,
        TextBlock {
            rect,
            text: b.text.clone(),
            chars_backing,
            chars_source,
            chars_cache: OnceLock::new(),
        },
    )
}

#[pymethods]
impl TextBlock {
    #[new]
    fn new(rect: Rectangle, text: String) -> Self {
        Self {
            rect,
            text,
            chars_backing: None,
            chars_source: BlockCharSource::None,
            chars_cache: OnceLock::new(),
        }
    }

    #[getter]
    fn chars(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let chars = self.chars_cache.get_or_init(|| self.materialize_chars());
        let out = PyList::empty(py);
        for ch in chars {
            out.append(char_to_py_object(py, ch)?)?;
        }
        Ok(out.into_any().unbind())
    }

    fn __repr__(&self) -> String {
        format!("TextBlock({:?})", self.text)
    }

    fn __eq__(&self, other: &TextBlock) -> bool {
        self.rect == other.rect && self.text == other.text
    }
}

impl TextBlock {
    fn materialize_chars(&self) -> Vec<CharBBox> {
        let Some(backing) = self.chars_backing.as_ref() else {
            return Vec::new();
        };
        match &self.chars_source {
            BlockCharSource::None => Vec::new(),
            BlockCharSource::Single(index) => backing.get(*index).cloned().unwrap_or_default(),
            BlockCharSource::Multi(indices) => {
                let total = indices
                    .iter()
                    .map(|index| backing.get(*index).map_or(0, Vec::len))
                    .sum();
                let mut out = Vec::with_capacity(total);
                for index in indices {
                    if let Some(chars) = backing.get(*index) {
                        out.extend_from_slice(chars);
                    }
                }
                out
            }
        }
    }
}

#[pymethods]
impl TextChar {
    fn __repr__(&self) -> String {
        format!("TextChar(ch={:?}, rect={})", self.ch, self.rect.__repr__())
    }
}

#[pymodule]
fn reap(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<TextBlockIndex>()?;
    m.add_class::<Rectangle>()?;
    m.add_class::<Point>()?;
    m.add_class::<TextBlock>()?;
    m.add_class::<TextChar>()?;
    m.add(
        "__all__",
        vec![
            "TextBlockIndex",
            "Rectangle",
            "Point",
            "TextBlock",
            "TextChar",
        ],
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::{BlockCharSource, Rectangle, TextBlock};
    use crate::text::CharBBox;

    fn make_char(ch: char, left: f64, right: f64) -> CharBBox {
        CharBBox {
            page_index: 0,
            ch,
            bbox: crate::text::Rectangle {
                top: 0.0,
                left,
                bottom: 10.0,
                right,
            },
        }
    }

    #[test]
    fn rectangle_with_coords_replaces_only_specified_values() {
        let rect = Rectangle::new(10.0, 20.0, 30.0, 40.0);
        let updated = rect.with_coords(Some(11.0), None, None, Some(44.0));
        assert!(updated == Rectangle::new(11.0, 20.0, 30.0, 44.0));
        assert!(rect == Rectangle::new(10.0, 20.0, 30.0, 40.0));
    }

    #[test]
    fn rectangle_side_helpers_chain() {
        let rect = Rectangle::new(5.0, 10.0, 15.0, 20.0);
        let updated = rect.with_bottom(50.0).with_right(25.0);
        assert!(updated == Rectangle::new(5.0, 10.0, 50.0, 25.0));
        assert!(rect == Rectangle::new(5.0, 10.0, 15.0, 20.0));
    }

    #[test]
    fn rectangle_with_margin_expands_normalized_rect() {
        let rect = Rectangle::new(10.0, 20.0, 30.0, 40.0);
        let updated = rect.with_margin(1.0, 2.0, 3.0, 4.0);
        assert!(updated == Rectangle::new(9.0, 17.0, 32.0, 44.0));
    }

    #[test]
    fn rectangle_with_margin_normalizes_before_applying_margins() {
        let rect = Rectangle::new(30.0, 40.0, 10.0, 20.0);
        let updated = rect.with_margin(1.0, 2.0, 3.0, 4.0);
        assert!(updated == Rectangle::new(9.0, 17.0, 32.0, 44.0));
    }

    #[test]
    fn rectangle_with_margin_negative_values_shrink() {
        let rect = Rectangle::new(0.0, 0.0, 10.0, 20.0);
        let updated = rect.with_margin(-1.0, -2.0, -3.0, -4.0);
        assert!(updated == Rectangle::new(1.0, 3.0, 8.0, 16.0));
    }

    #[test]
    fn rectangle_invalid_outputs_are_allowed_and_detectable_with_validate() {
        let rect = Rectangle::new(0.0, 0.0, 10.0, 10.0);
        let updated = rect.with_top(11.0);
        assert!(updated == Rectangle::new(11.0, 0.0, 10.0, 10.0));
        assert!(!updated.validate());
    }

    #[test]
    fn rectangle_expand_returns_expected_result_for_simple_case() {
        let rect = Rectangle::new(10.0, 10.0, 20.0, 20.0);
        let blockers = vec![Rectangle::new(0.0, 24.0, 100.0, 80.0)];
        let max_bounds = Rectangle::new(0.0, 0.0, 100.0, 100.0);
        let expanded = rect
            .expand(blockers, vec!["right".to_string()], max_bounds)
            .expect("expand should succeed");
        assert!(expanded == Rectangle::new(10.0, 10.0, 20.0, 24.0));
        assert!(rect == Rectangle::new(10.0, 10.0, 20.0, 20.0));
    }

    #[test]
    fn rectangle_expand_rejects_invalid_direction() {
        let rect = Rectangle::new(10.0, 10.0, 20.0, 20.0);
        let max_bounds = Rectangle::new(0.0, 0.0, 100.0, 100.0);
        let err = rect
            .expand(vec![], vec!["north".to_string()], max_bounds)
            .expect_err("expand should fail");
        assert_eq!(
            err.to_string(),
            "ValueError: invalid direction 'north'; expected one of: up, down, left, right"
        );
    }

    #[test]
    fn rectangle_expand_surfaces_maximum_bounds_errors() {
        let rect = Rectangle::new(10.0, 10.0, 20.0, 20.0);
        let max_bounds = Rectangle::new(12.0, 12.0, 18.0, 18.0);
        let err = rect
            .expand(vec![], vec!["right".to_string()], max_bounds)
            .expect_err("expand should fail");
        assert_eq!(
            err.to_string(),
            "ValueError: maximum_bounds must fully contain the rectangle being expanded"
        );
    }

    #[test]
    fn text_block_materialize_chars_uses_single_source_without_copying_spaces() {
        let backing = Arc::new(vec![
            vec![make_char('A', 0.0, 5.0), make_char('B', 5.0, 10.0)],
            vec![make_char('C', 10.0, 15.0)],
        ]);
        let block = TextBlock {
            rect: Rectangle::new(0.0, 0.0, 10.0, 10.0),
            text: "AB".to_string(),
            chars_backing: Some(backing),
            chars_source: BlockCharSource::Single(0),
            chars_cache: std::sync::OnceLock::new(),
        };
        let text = block
            .materialize_chars()
            .iter()
            .map(|ch| ch.ch)
            .collect::<String>();
        assert_eq!(text, "AB");
    }

    #[test]
    fn text_block_materialize_chars_merges_sources_in_order_without_synthetic_space_chars() {
        let backing = Arc::new(vec![
            vec![make_char('A', 0.0, 5.0), make_char('B', 5.0, 10.0)],
            vec![make_char('C', 12.0, 17.0), make_char('D', 17.0, 22.0)],
        ]);
        let block = TextBlock {
            rect: Rectangle::new(0.0, 0.0, 10.0, 22.0),
            text: "AB CD".to_string(),
            chars_backing: Some(backing),
            chars_source: BlockCharSource::Multi(vec![0, 1]),
            chars_cache: std::sync::OnceLock::new(),
        };
        let chars = block.materialize_chars();
        let text = chars.iter().map(|ch| ch.ch).collect::<String>();
        assert_eq!(text, "ABCD");
    }
}
