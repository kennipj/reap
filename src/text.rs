use std::collections::HashMap;
use std::fs;
use std::sync::{Arc, OnceLock, RwLock};

use crate::model::Object;
use crate::parser::PdfDoc;
use crate::tokenizer::{Lexer, Token};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Point {
    pub x: f64,
    pub y: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rectangle {
    pub top: f64,
    pub left: f64,
    pub bottom: f64,
    pub right: f64,
}

impl Rectangle {
    fn normalized(&self) -> (f64, f64, f64, f64) {
        let left = self.left.min(self.right);
        let right = self.left.max(self.right);
        let top = self.top.min(self.bottom);
        let bottom = self.top.max(self.bottom);
        (left, top, right, bottom)
    }

    pub fn get_center(&self) -> (f64, f64) {
        (
            (self.left + self.right) / 2.0,
            (self.top + self.bottom) / 2.0,
        )
    }

    pub fn overlap_percentage(&self, other: &Rectangle) -> f64 {
        let (l1, t1, r1, b1) = self.normalized();
        let (l2, t2, r2, b2) = other.normalized();
        let x_overlap = (r1.min(r2) - l1.max(l2)).max(0.0);
        let y_overlap = (b1.min(b2) - t1.max(t2)).max(0.0);
        let overlap_area = x_overlap * y_overlap;
        let self_area = (r1 - l1) * (b1 - t1);
        if self_area > 0.0 {
            overlap_area / self_area
        } else {
            0.0
        }
    }

    pub fn overlaps_horizontally(&self, other: &Rectangle) -> bool {
        let (l1, _, r1, _) = self.normalized();
        let (l2, _, r2, _) = other.normalized();
        l1.max(l2) < r1.min(r2)
    }

    pub fn contains_left_side(&self, other: &Rectangle) -> bool {
        let (l1, t1, _, b1) = self.normalized();
        let (l2, t2, _, b2) = other.normalized();
        l1 <= l2 + 0.2 && t1 <= t2 + 0.2 && b1 >= b2 - 0.2
    }

    pub fn overlaps(&self, other: &Rectangle, pct: f64) -> bool {
        let (l1, t1, r1, b1) = self.normalized();
        let (l2, t2, r2, b2) = other.normalized();
        if r1 <= l2 || r2 <= l1 || b1 <= t2 || b2 <= t1 {
            return false;
        }
        self.overlap_percentage(other) >= pct
    }

    pub fn validate(&self) -> bool {
        let (l, t, r, b) = self.normalized();
        t < b && l < r
    }

    pub fn overlaps_with_margins(
        &self,
        other: &Rectangle,
        top: f64,
        bottom: f64,
        left: f64,
        right: f64,
    ) -> bool {
        let (l, t, r, b) = self.normalized();
        let expanded = Rectangle {
            top: t - top,
            left: l - left,
            bottom: b + bottom,
            right: r + right,
        };
        other.overlaps(&expanded, 0.0)
    }

    pub fn vertical_overlap(&self, other: &Rectangle, threshold: f64) -> bool {
        let (_, t1, _, b1) = self.normalized();
        let (_, t2, _, b2) = other.normalized();
        let overlap_top = t1.max(t2);
        let overlap_bottom = b1.min(b2);
        let overlap_height = (overlap_bottom - overlap_top).max(0.0);
        let height = b1 - t1;
        if height <= 0.0 {
            return false;
        }
        (overlap_height / height) > threshold
    }

    pub fn distance(&self, other: &Rectangle) -> f64 {
        let (cx1, cy1) = self.get_center();
        let (cx2, cy2) = other.get_center();
        ((cx1 - cx2).powi(2) + (cy1 - cy2).powi(2)).sqrt()
    }

    pub fn corner(&self, left_right: LeftRight, top_bottom: TopBottom) -> Point {
        let x = match left_right {
            LeftRight::Left => self.left,
            LeftRight::Right => self.right,
        };
        let y = match top_bottom {
            TopBottom::Top => self.top,
            TopBottom::Bottom => self.bottom,
        };
        Point { x, y }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum LeftRight {
    Left,
    Right,
}

#[derive(Debug, Clone, Copy)]
pub enum TopBottom {
    Top,
    Bottom,
}

#[derive(Debug, Clone)]
pub struct CharBBox {
    pub page_index: usize,
    pub ch: char,
    pub bbox: Rectangle,
}

#[derive(Debug, Clone)]
pub struct TextBlock {
    pub page_index: usize,
    pub text: String,
    pub bbox: Rectangle,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TextStyle {
    pub font_size: f64,
    pub fill_rgb: Option<(f64, f64, f64)>,
    pub font_name: Option<String>,
}

#[derive(Clone, Copy, Debug)]
struct Matrix {
    a: f64,
    b: f64,
    c: f64,
    d: f64,
    e: f64,
    f: f64,
}

impl Matrix {
    fn identity() -> Self {
        Self {
            a: 1.0,
            b: 0.0,
            c: 0.0,
            d: 1.0,
            e: 0.0,
            f: 0.0,
        }
    }

    fn multiply(self, other: Matrix) -> Matrix {
        Matrix {
            a: self.a * other.a + self.c * other.b,
            b: self.b * other.a + self.d * other.b,
            c: self.a * other.c + self.c * other.d,
            d: self.b * other.c + self.d * other.d,
            e: self.a * other.e + self.c * other.f + self.e,
            f: self.b * other.e + self.d * other.f + self.f,
        }
    }

    fn translate(tx: f64, ty: f64) -> Matrix {
        Matrix {
            a: 1.0,
            b: 0.0,
            c: 0.0,
            d: 1.0,
            e: tx,
            f: ty,
        }
    }

    fn apply_to_point(self, x: f64, y: f64) -> (f64, f64) {
        (
            x * self.a + y * self.c + self.e,
            x * self.b + y * self.d + self.f,
        )
    }
}

#[derive(Debug, Clone)]
struct FontMetrics {
    first_char: i64,
    widths: Vec<f64>,
    ascent: f64,
    descent: f64,
    units_to_text: f64,
    to_unicode: Option<CMap>,
    encoding: Option<EncodingMap>,
    single_byte_encoding: SingleByteEncoding,
    single_byte_overrides: HashMap<u32, char>,
    cid_widths: Vec<CidWidthRange>,
    cid_default_width: i64,
}

#[derive(Debug, Clone, Copy)]
enum SingleByteEncoding {
    WinAnsi,
    MacRoman,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ColorSpaceKind {
    DeviceGray,
    DeviceRgb,
    DeviceCmyk,
    Unknown,
}

impl FontMetrics {
    fn width(&self, code: u8) -> f64 {
        let idx = code as i64 - self.first_char;
        if idx >= 0 && (idx as usize) < self.widths.len() {
            self.widths[idx as usize]
        } else {
            0.0
        }
    }

    fn width_cid(&self, code: u32) -> f64 {
        for range in &self.cid_widths {
            if code >= range.start && code <= range.end {
                return range.width as f64;
            }
        }
        self.cid_default_width as f64
    }
}

const TYPE3_TEXT_HEIGHT_MIN: f64 = 0.2;
const TYPE3_TEXT_HEIGHT_MAX: f64 = 3.0;
const TYPE3_FALLBACK_TEXT_HEIGHT: f64 = 1.0;
const TYPE3_UNITS_EPSILON: f64 = 1e-9;
const MAX_TRACKED_PAINTED_FILLS: usize = 4096;
const MIN_TEXT_BACKGROUND_CONTRAST_RATIO: f64 = 1.05;

#[derive(Debug, Clone)]
struct TextState {
    font: Option<String>,
    font_size: f64,
    char_spacing: f64,
    word_spacing: f64,
    horiz_scaling: f64,
    leading: f64,
    rise: f64,
    render_mode: i64,
    fill_alpha: Option<f64>,
    stroke_alpha: Option<f64>,
    fill_rgb: Option<(f64, f64, f64)>,
    fill_color_space: ColorSpaceKind,
    text_matrix: Matrix,
    line_matrix: Matrix,
}

#[derive(Clone, Copy, Debug)]
struct Rect {
    min_x: f64,
    min_y: f64,
    max_x: f64,
    max_y: f64,
}

#[derive(Clone, Copy, Debug)]
struct PaintedFill {
    bbox: Rect,
    bg_luminance: Option<f64>,
}

impl Rect {
    fn from_points(points: &[(f64, f64)]) -> Option<Self> {
        if points.is_empty() {
            return None;
        }
        let mut min_x = points[0].0;
        let mut max_x = points[0].0;
        let mut min_y = points[0].1;
        let mut max_y = points[0].1;
        for (x, y) in points.iter().copied() {
            min_x = min_x.min(x);
            max_x = max_x.max(x);
            min_y = min_y.min(y);
            max_y = max_y.max(y);
        }
        Some(Self {
            min_x,
            min_y,
            max_x,
            max_y,
        })
    }

    fn intersects(&self, other: &Rect) -> bool {
        self.min_x <= other.max_x
            && self.max_x >= other.min_x
            && self.min_y <= other.max_y
            && self.max_y >= other.min_y
    }

    fn intersect(self, other: Rect) -> Rect {
        Rect {
            min_x: self.min_x.max(other.min_x),
            min_y: self.min_y.max(other.min_y),
            max_x: self.max_x.min(other.max_x),
            max_y: self.max_y.min(other.max_y),
        }
    }

    fn is_empty(&self) -> bool {
        self.min_x >= self.max_x || self.min_y >= self.max_y
    }

    fn union(self, other: Rect) -> Rect {
        Rect {
            min_x: self.min_x.min(other.min_x),
            min_y: self.min_y.min(other.min_y),
            max_x: self.max_x.max(other.max_x),
            max_y: self.max_y.max(other.max_y),
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct PageLayout {
    x_min: f64,
    y_min: f64,
    width: f64,
    height: f64,
    rotation: i32,
    rotated_width: f64,
    rotated_height: f64,
    y_offset: f64,
}

impl TextState {
    fn new() -> Self {
        Self {
            font: None,
            font_size: 12.0,
            char_spacing: 0.0,
            word_spacing: 0.0,
            horiz_scaling: 100.0,
            leading: 0.0,
            rise: 0.0,
            render_mode: 0,
            fill_alpha: None,
            stroke_alpha: None,
            fill_rgb: None,
            fill_color_space: ColorSpaceKind::DeviceGray,
            text_matrix: Matrix::identity(),
            line_matrix: Matrix::identity(),
        }
    }
}

type FontMapCache = HashMap<usize, Arc<HashMap<String, FontMetrics>>>;

pub fn extract_char_bboxes(doc: &PdfDoc) -> Vec<CharBBox> {
    let pages = collect_pages(doc);
    let layouts = page_layouts_for_pages(doc, &pages);
    let raw = extract_char_bboxes_raw_from_pages(doc, &pages);
    canonicalize_char_bboxes(raw, &layouts)
}

fn extract_char_bboxes_raw_from_pages(doc: &PdfDoc, pages: &[Object]) -> Vec<CharBBox> {
    let mut out = Vec::new();
    let mut font_map_cache: FontMapCache = HashMap::new();
    for (page_index, page) in pages.iter().enumerate() {
        let (resources, contents) = page_resources_and_contents(doc, page);
        let content_bytes = decode_contents(doc, &contents);
        let ctm = Matrix::identity();
        let text_state = TextState::new();
        process_content(
            doc,
            page_index,
            resources,
            &content_bytes,
            ctm,
            text_state,
            &mut font_map_cache,
            &mut out,
        );
        for (bytes, ann_resources, ann_matrix) in annotation_appearance_streams(doc, page) {
            let text_state = TextState::new();
            process_content(
                doc,
                page_index,
                ann_resources.or(resources),
                &bytes,
                ann_matrix,
                text_state,
                &mut font_map_cache,
                &mut out,
            );
        }
    }
    out
}

fn flush_current_text_block(
    out_blocks: &mut Vec<TextBlock>,
    cur_text: &mut String,
    cur_page: &mut Option<usize>,
    cur_bbox: &mut Option<Rect>,
) {
    if cur_text.is_empty() {
        return;
    }
    if is_discardable_underscore_block(cur_text) {
        cur_text.clear();
        *cur_page = None;
        *cur_bbox = None;
        return;
    }
    if let (Some(page), Some(b)) = (cur_page.take(), cur_bbox.take()) {
        out_blocks.push(TextBlock {
            page_index: page,
            text: std::mem::take(cur_text),
            bbox: Rectangle {
                left: b.min_x,
                top: b.min_y,
                right: b.max_x,
                bottom: b.max_y,
            },
        });
    } else {
        cur_text.clear();
        *cur_page = None;
        *cur_bbox = None;
    }
}

fn is_discardable_underscore_block(text: &str) -> bool {
    !text.is_empty() && text.chars().all(|ch| ch == '_')
}

const WORD_POS_GAP_HEIGHT_RATIO: f64 = 0.10;
const WORD_POS_GAP_MIN: f64 = 0.5;
const WORD_NEG_GAP_HEIGHT_RATIO: f64 = 2.0;

fn word_gap_thresholds(prev_height: f64) -> (f64, f64) {
    let pos_gap_threshold = (prev_height * WORD_POS_GAP_HEIGHT_RATIO).max(WORD_POS_GAP_MIN);
    let neg_gap_threshold = prev_height * WORD_NEG_GAP_HEIGHT_RATIO;
    (pos_gap_threshold, neg_gap_threshold)
}

#[derive(Default)]
struct BlockAccumulator {
    out_blocks: Vec<TextBlock>,
    cur_text: String,
    cur_page: Option<usize>,
    cur_bbox: Option<Rect>,
    prev_bbox: Option<Rect>,
    prev_font_size: Option<f64>,
    prev_ch: Option<char>,
}

impl BlockAccumulator {
    fn push_whitespace(&mut self) {
        flush_current_text_block(
            &mut self.out_blocks,
            &mut self.cur_text,
            &mut self.cur_page,
            &mut self.cur_bbox,
        );
        self.prev_bbox = None;
        self.prev_font_size = None;
        self.prev_ch = None;
    }

    fn push_char(&mut self, ch: CharBBox, font_size: f64) {
        let bbox = Rect {
            min_x: ch.bbox.left,
            min_y: ch.bbox.top,
            max_x: ch.bbox.right,
            max_y: ch.bbox.bottom,
        };
        let mut split = false;
        if let Some(prev) = self.prev_bbox {
            if self.cur_page != Some(ch.page_index) {
                split = true;
            } else {
                let prev_h = (prev.max_y - prev.min_y).abs();
                let y_delta = (bbox.min_y - prev.min_y).abs();
                let gap = bbox.min_x - prev.max_x;
                let line_threshold = prev_h * 0.5;
                let cur_h = (bbox.max_y - bbox.min_y).abs();
                let (pos_gap_threshold, neg_gap_threshold) = word_gap_thresholds(prev_h);
                if prev_h > 0.0 {
                    let ratio = cur_h / prev_h;
                    if ratio > 1.35 || ratio < 0.75 {
                        split = true;
                    }
                }
                if y_delta > line_threshold {
                    split = true;
                } else if gap > pos_gap_threshold || gap < -neg_gap_threshold {
                    split = true;
                } else if gap > WORD_POS_GAP_MIN
                    && self.prev_ch.is_some_and(|prev| prev.is_lowercase())
                    && ch.ch.is_uppercase()
                {
                    // Preserve intended boundaries in compact labels like "Tips Reported"
                    // when spacing exists but no explicit whitespace character is present.
                    split = true;
                }
            }
        }
        if let Some(prev_font_size) = self.prev_font_size {
            if (prev_font_size - font_size).abs() > prev_font_size * 0.05 {
                split = true;
            }
        }

        if split && !self.cur_text.is_empty() {
            flush_current_text_block(
                &mut self.out_blocks,
                &mut self.cur_text,
                &mut self.cur_page,
                &mut self.cur_bbox,
            );
        }

        if self.cur_text.is_empty() {
            self.cur_text.push(ch.ch);
            self.cur_page = Some(ch.page_index);
            self.cur_bbox = Some(bbox);
        } else if self.cur_page != Some(ch.page_index) {
            flush_current_text_block(
                &mut self.out_blocks,
                &mut self.cur_text,
                &mut self.cur_page,
                &mut self.cur_bbox,
            );
            self.cur_text.push(ch.ch);
            self.cur_page = Some(ch.page_index);
            self.cur_bbox = Some(bbox);
        } else if let Some(existing) = self.cur_bbox {
            self.cur_text.push(ch.ch);
            self.cur_bbox = Some(existing.union(bbox));
        }

        self.prev_bbox = Some(bbox);
        self.prev_font_size = Some(font_size);
        self.prev_ch = Some(ch.ch);
    }

    fn finish(mut self) -> Vec<TextBlock> {
        flush_current_text_block(
            &mut self.out_blocks,
            &mut self.cur_text,
            &mut self.cur_page,
            &mut self.cur_bbox,
        );
        self.out_blocks
    }
}

fn extract_text_blocks_raw(doc: &PdfDoc, pages: &[Object]) -> Vec<TextBlock> {
    let mut blocks = BlockAccumulator::default();
    let mut font_map_cache: FontMapCache = HashMap::new();
    for (page_index, page) in pages.iter().enumerate() {
        let (resources, contents) = page_resources_and_contents(doc, page);
        let content_bytes = decode_contents(doc, &contents);
        let ctm = Matrix::identity();
        let text_state = TextState::new();
        process_content_with_blocks(
            doc,
            page_index,
            resources,
            &content_bytes,
            ctm,
            text_state,
            &mut font_map_cache,
            &mut blocks,
        );
        for (bytes, ann_resources, ann_matrix) in annotation_appearance_streams(doc, page) {
            let text_state = TextState::new();
            process_content_with_blocks(
                doc,
                page_index,
                ann_resources.or(resources),
                &bytes,
                ann_matrix,
                text_state,
                &mut font_map_cache,
                &mut blocks,
            );
        }
    }
    blocks.finish()
}

pub fn extract_text_blocks(doc: &PdfDoc) -> Vec<TextBlock> {
    let pages = collect_pages(doc);
    let layouts = page_layouts_for_pages(doc, &pages);
    let raw = extract_text_blocks_raw(doc, &pages);
    canonicalize_text_blocks(raw, &layouts)
}

pub fn extract_char_bboxes_with_style(doc: &PdfDoc) -> Vec<(CharBBox, TextStyle)> {
    let pages = collect_pages(doc);
    let layouts = page_layouts_for_pages(doc, &pages);
    let raw = extract_char_bboxes_with_style_raw_from_pages(doc, &pages);
    canonicalize_char_bboxes_with_style(raw, &layouts)
}

fn extract_char_bboxes_with_style_raw_from_pages(
    doc: &PdfDoc,
    pages: &[Object],
) -> Vec<(CharBBox, TextStyle)> {
    let mut out = Vec::new();
    let mut font_map_cache: FontMapCache = HashMap::new();
    for (page_index, page) in pages.iter().enumerate() {
        let (resources, contents) = page_resources_and_contents(doc, page);
        let content_bytes = decode_contents(doc, &contents);
        let ctm = Matrix::identity();
        let text_state = TextState::new();
        process_content_with_style(
            doc,
            page_index,
            resources,
            &content_bytes,
            ctm,
            text_state,
            &mut font_map_cache,
            &mut out,
        );
        for (bytes, ann_resources, ann_matrix) in annotation_appearance_streams(doc, page) {
            let text_state = TextState::new();
            process_content_with_style(
                doc,
                page_index,
                ann_resources.or(resources),
                &bytes,
                ann_matrix,
                text_state,
                &mut font_map_cache,
                &mut out,
            );
        }
    }
    out
}

fn process_content(
    doc: &PdfDoc,
    page_index: usize,
    resources: Option<&Object>,
    content_bytes: &[u8],
    ctm: Matrix,
    text_state: TextState,
    font_map_cache: &mut FontMapCache,
    out: &mut Vec<CharBBox>,
) {
    let mut painted_fills = Vec::new();
    process_content_impl(
        doc,
        page_index,
        resources,
        content_bytes,
        ctm,
        text_state,
        &mut painted_fills,
        font_map_cache,
        out,
        show_text,
    );
}

fn process_content_with_style(
    doc: &PdfDoc,
    page_index: usize,
    resources: Option<&Object>,
    content_bytes: &[u8],
    ctm: Matrix,
    text_state: TextState,
    font_map_cache: &mut FontMapCache,
    out: &mut Vec<(CharBBox, TextStyle)>,
) {
    let mut painted_fills = Vec::new();
    process_content_impl(
        doc,
        page_index,
        resources,
        content_bytes,
        ctm,
        text_state,
        &mut painted_fills,
        font_map_cache,
        out,
        show_text_with_style,
    );
}

fn process_content_with_blocks(
    doc: &PdfDoc,
    page_index: usize,
    resources: Option<&Object>,
    content_bytes: &[u8],
    ctm: Matrix,
    text_state: TextState,
    font_map_cache: &mut FontMapCache,
    out: &mut BlockAccumulator,
) {
    let mut painted_fills = Vec::new();
    process_content_impl(
        doc,
        page_index,
        resources,
        content_bytes,
        ctm,
        text_state,
        &mut painted_fills,
        font_map_cache,
        out,
        show_text_for_block_accumulator,
    );
}

fn process_content_impl<C, F>(
    doc: &PdfDoc,
    page_index: usize,
    resources: Option<&Object>,
    content_bytes: &[u8],
    mut ctm: Matrix,
    text_state: TextState,
    painted_fills: &mut Vec<PaintedFill>,
    font_map_cache: &mut FontMapCache,
    out: &mut C,
    emit_text: F,
) where
    F: Fn(
            usize,
            &HashMap<String, FontMetrics>,
            &mut TextState,
            &Matrix,
            Option<Rect>,
            &[PaintedFill],
            &[u8],
            &mut C,
        ) + Copy,
{
    let mut text_state = text_state;
    let mut ctm_stack: Vec<Matrix> = Vec::new();
    let mut state_stack: Vec<TextState> = Vec::new();
    let mut clip_stack: Vec<Option<Rect>> = Vec::new();
    let mut current_clip: Option<Rect> = None;
    let mut path_points: Vec<(f64, f64)> = Vec::new();
    let mut path_bbox: Option<Rect> = None;
    let mut tokenizer = ContentTokenizer::new(content_bytes);
    let mut operands: Vec<Object> = Vec::with_capacity(8);
    let font_map_key = resources.map_or(0usize, |r| r as *const Object as usize);
    let font_map = if let Some(cached) = font_map_cache.get(&font_map_key) {
        cached.clone()
    } else {
        let built = Arc::new(build_font_map(doc, resources));
        font_map_cache.insert(font_map_key, built.clone());
        built
    };
    let resources_dict = resources.and_then(|r| r.as_dict());
    let xobject_dict = resources_dict
        .and_then(|d| d.get("XObject"))
        .and_then(|v| doc.resolve(v).as_dict());
    let extgstate_dict = resources_dict
        .and_then(|d| d.get("ExtGState"))
        .and_then(|v| doc.resolve(v).as_dict());

    while let Some(op) = tokenizer.next_op_into(&mut operands) {
        match op.as_str() {
            "cs" => {
                if let Some(Object::Name(name)) = operands.first() {
                    text_state.fill_color_space =
                        resolve_non_stroking_color_space_kind(doc, resources_dict, name);
                }
            }
            "g" | "rg" | "k" | "sc" | "scn" => {
                if let Some(rgb) =
                    parse_non_stroking_rgb(op.as_str(), &operands, text_state.fill_color_space)
                {
                    text_state.fill_rgb = Some(rgb);
                }
                if op == "g" {
                    text_state.fill_color_space = ColorSpaceKind::DeviceGray;
                } else if op == "rg" {
                    text_state.fill_color_space = ColorSpaceKind::DeviceRgb;
                } else if op == "k" {
                    text_state.fill_color_space = ColorSpaceKind::DeviceCmyk;
                }
            }
            "BMC" | "BDC" => {}
            "EMC" => {}
            "q" => {
                ctm_stack.push(ctm);
                state_stack.push(text_state.clone());
                clip_stack.push(current_clip);
            }
            "Q" => {
                if let Some(prev) = ctm_stack.pop() {
                    ctm = prev;
                }
                if let Some(prev) = state_stack.pop() {
                    text_state = prev;
                }
                if let Some(prev) = clip_stack.pop() {
                    current_clip = prev;
                }
            }
            "cm" => {
                if operands.len() == 6 {
                    let m = Matrix {
                        a: num(&operands[0]),
                        b: num(&operands[1]),
                        c: num(&operands[2]),
                        d: num(&operands[3]),
                        e: num(&operands[4]),
                        f: num(&operands[5]),
                    };
                    // PDF cm concatenates onto the current CTM; post-multiply so clip and text stay aligned.
                    ctm = ctm.multiply(m);
                }
            }
            "m" => {
                if operands.len() == 2 {
                    let x = num(&operands[0]);
                    let y = num(&operands[1]);
                    let p = ctm.apply_to_point(x, y);
                    path_points.clear();
                    path_points.push(p);
                }
            }
            "l" => {
                if operands.len() == 2 {
                    let x = num(&operands[0]);
                    let y = num(&operands[1]);
                    let p = ctm.apply_to_point(x, y);
                    path_points.push(p);
                }
            }
            "re" => {
                if operands.len() == 4 {
                    let x = num(&operands[0]);
                    let y = num(&operands[1]);
                    let w = num(&operands[2]);
                    let h = num(&operands[3]);
                    let p0 = ctm.apply_to_point(x, y);
                    let p1 = ctm.apply_to_point(x + w, y);
                    let p2 = ctm.apply_to_point(x + w, y + h);
                    let p3 = ctm.apply_to_point(x, y + h);
                    path_bbox = Rect::from_points(&[p0, p1, p2, p3]);
                    path_points.clear();
                }
            }
            "h" => {}
            "W" | "W*" => {
                let mut bbox = path_bbox.or_else(|| Rect::from_points(&path_points));
                if let Some(b) = bbox.take() {
                    current_clip = Some(if let Some(existing) = current_clip {
                        existing.intersect(b)
                    } else {
                        b
                    });
                }
            }
            "f" | "F" | "f*" | "B" | "B*" | "b" | "b*" => {
                capture_painted_fill(
                    path_bbox,
                    &path_points,
                    current_clip,
                    &text_state,
                    painted_fills,
                );
                path_points.clear();
                path_bbox = None;
            }
            "n" | "S" | "s" => {
                path_points.clear();
                path_bbox = None;
            }
            "gs" => {
                if let Some(Object::Name(name)) = operands.get(0) {
                    if let Some(gs_dict) = extgstate_dict {
                        if let Some(state_obj) = gs_dict.get(name) {
                            if let Some(state) = doc.resolve(state_obj).as_dict() {
                                let tr = state
                                    .get("Tr")
                                    .or_else(|| state.get("TR"))
                                    .or_else(|| state.get("TR2"))
                                    .and_then(|v| v.as_i64());
                                if let Some(v) = tr {
                                    text_state.render_mode = v;
                                }
                                if let Some(v) = state.get("ca").and_then(|v| v.as_f64()) {
                                    text_state.fill_alpha = Some(v);
                                }
                                if let Some(v) = state.get("CA").and_then(|v| v.as_f64()) {
                                    text_state.stroke_alpha = Some(v);
                                }
                            }
                        }
                    }
                }
            }
            "Do" => {
                if let Some(Object::Name(name)) = operands.get(0) {
                    if let Some(xobj_dict) = xobject_dict {
                        if let Some(xobj) = xobj_dict.get(name) {
                            let xobj = doc.resolve(xobj);
                            if let Object::Stream { dict, .. } = xobj {
                                let subtype = dict.get("Subtype").and_then(|v| v.as_name());
                                if subtype == Some("Form") {
                                    let form_matrix = dict
                                        .get("Matrix")
                                        .and_then(|v| doc.resolve(v).as_array())
                                        .and_then(|arr| {
                                            if arr.len() == 6 {
                                                Some(Matrix {
                                                    a: arr[0].as_f64().unwrap_or(1.0),
                                                    b: arr[1].as_f64().unwrap_or(0.0),
                                                    c: arr[2].as_f64().unwrap_or(0.0),
                                                    d: arr[3].as_f64().unwrap_or(1.0),
                                                    e: arr[4].as_f64().unwrap_or(0.0),
                                                    f: arr[5].as_f64().unwrap_or(0.0),
                                                })
                                            } else {
                                                None
                                            }
                                        })
                                        .unwrap_or_else(Matrix::identity);
                                    let form_resources =
                                        dict.get("Resources").map(|r| doc.resolve(r));
                                    if let Some(form_bytes) = decode_stream(doc, xobj) {
                                        process_content_impl(
                                            doc,
                                            page_index,
                                            form_resources.or(resources),
                                            &form_bytes,
                                            ctm.multiply(form_matrix),
                                            text_state.clone(),
                                            painted_fills,
                                            font_map_cache,
                                            out,
                                            emit_text,
                                        );
                                    }
                                }
                            }
                        }
                    }
                }
            }
            "BT" => {
                text_state.text_matrix = Matrix::identity();
                text_state.line_matrix = Matrix::identity();
            }
            "ET" => {}
            "Tf" => {
                if operands.len() >= 2 {
                    if let Some(name) = operands[0].as_name() {
                        text_state.font = Some(name.to_string());
                    }
                    text_state.font_size = num(&operands[1]);
                }
            }
            "Tm" => {
                if operands.len() == 6 {
                    let m = Matrix {
                        a: num(&operands[0]),
                        b: num(&operands[1]),
                        c: num(&operands[2]),
                        d: num(&operands[3]),
                        e: num(&operands[4]),
                        f: num(&operands[5]),
                    };
                    text_state.text_matrix = m;
                    text_state.line_matrix = m;
                }
            }
            "Td" => {
                if operands.len() == 2 {
                    let tx = num(&operands[0]);
                    let ty = num(&operands[1]);
                    text_state.line_matrix =
                        text_state.line_matrix.multiply(Matrix::translate(tx, ty));
                    text_state.text_matrix = text_state.line_matrix;
                }
            }
            "TD" => {
                if operands.len() == 2 {
                    let ty = num(&operands[1]);
                    text_state.leading = -ty;
                    let tx = num(&operands[0]);
                    text_state.line_matrix =
                        text_state.line_matrix.multiply(Matrix::translate(tx, ty));
                    text_state.text_matrix = text_state.line_matrix;
                }
            }
            "T*" => {
                let ty = -text_state.leading;
                text_state.line_matrix =
                    text_state.line_matrix.multiply(Matrix::translate(0.0, ty));
                text_state.text_matrix = text_state.line_matrix;
            }
            "TL" => {
                if let Some(v) = operands.get(0) {
                    text_state.leading = num(v);
                }
            }
            "Tc" => {
                if let Some(v) = operands.get(0) {
                    text_state.char_spacing = num(v);
                }
            }
            "Tw" => {
                if let Some(v) = operands.get(0) {
                    text_state.word_spacing = num(v);
                }
            }
            "Tz" => {
                if let Some(v) = operands.get(0) {
                    text_state.horiz_scaling = num(v);
                }
            }
            "Ts" => {
                if let Some(v) = operands.get(0) {
                    text_state.rise = num(v);
                }
            }
            "Tr" => {
                if let Some(v) = operands.get(0) {
                    text_state.render_mode = v.as_i64().unwrap_or(0);
                }
            }
            "'" => {
                let ty = -text_state.leading;
                text_state.line_matrix =
                    text_state.line_matrix.multiply(Matrix::translate(0.0, ty));
                text_state.text_matrix = text_state.line_matrix;
                if let Some(Object::String(bytes)) = operands.get(0) {
                    emit_text(
                        page_index,
                        font_map.as_ref(),
                        &mut text_state,
                        &ctm,
                        current_clip,
                        painted_fills,
                        bytes,
                        out,
                    );
                }
            }
            "\"" => {
                if operands.len() >= 3 {
                    text_state.word_spacing = num(&operands[0]);
                    text_state.char_spacing = num(&operands[1]);
                    let ty = -text_state.leading;
                    text_state.line_matrix =
                        text_state.line_matrix.multiply(Matrix::translate(0.0, ty));
                    text_state.text_matrix = text_state.line_matrix;
                    if let Some(Object::String(bytes)) = operands.get(2) {
                        emit_text(
                            page_index,
                            font_map.as_ref(),
                            &mut text_state,
                            &ctm,
                            current_clip,
                            painted_fills,
                            bytes,
                            out,
                        );
                    }
                }
            }
            "Tj" => {
                if let Some(Object::String(bytes)) = operands.get(0) {
                    emit_text(
                        page_index,
                        font_map.as_ref(),
                        &mut text_state,
                        &ctm,
                        current_clip,
                        painted_fills,
                        bytes,
                        out,
                    );
                }
            }
            "TJ" => {
                if let Some(Object::Array(items)) = operands.get(0) {
                    for item in items {
                        match item {
                            Object::String(bytes) => {
                                emit_text(
                                    page_index,
                                    font_map.as_ref(),
                                    &mut text_state,
                                    &ctm,
                                    current_clip,
                                    painted_fills,
                                    bytes,
                                    out,
                                );
                            }
                            Object::Integer(kern) => {
                                let adjust = (*kern as f64 / 1000.0)
                                    * text_state.font_size
                                    * (text_state.horiz_scaling / 100.0);
                                text_state.text_matrix = text_state
                                    .text_matrix
                                    .multiply(Matrix::translate(-adjust, 0.0));
                            }
                            Object::Real(kern) => {
                                let adjust = (*kern / 1000.0)
                                    * text_state.font_size
                                    * (text_state.horiz_scaling / 100.0);
                                text_state.text_matrix = text_state
                                    .text_matrix
                                    .multiply(Matrix::translate(-adjust, 0.0));
                            }
                            _ => {}
                        }
                    }
                }
            }
            _ => {}
        }
    }
}

fn num(obj: &Object) -> f64 {
    obj.as_f64().unwrap_or(0.0)
}

fn clamp_unit_interval(v: f64) -> f64 {
    if !v.is_finite() {
        return 0.0;
    }
    v.clamp(0.0, 1.0)
}

fn operand_as_number(operands: &[Object], index: usize) -> Option<f64> {
    operands.get(index).and_then(|operand| operand.as_f64())
}

fn color_space_kind_from_name(name: &str) -> Option<ColorSpaceKind> {
    match name {
        "DeviceGray" | "G" | "CalGray" => Some(ColorSpaceKind::DeviceGray),
        "DeviceRGB" | "RGB" | "CalRGB" => Some(ColorSpaceKind::DeviceRgb),
        "DeviceCMYK" | "CMYK" => Some(ColorSpaceKind::DeviceCmyk),
        _ => None,
    }
}

fn color_space_kind_from_object(doc: &PdfDoc, obj: &Object) -> ColorSpaceKind {
    let resolved = doc.resolve(obj);
    if let Some(name) = resolved.as_name() {
        return color_space_kind_from_name(name).unwrap_or(ColorSpaceKind::Unknown);
    }
    let Some(arr) = resolved.as_array() else {
        return ColorSpaceKind::Unknown;
    };
    let Some(space_name) = arr.first().and_then(|v| doc.resolve(v).as_name()) else {
        return ColorSpaceKind::Unknown;
    };
    match space_name {
        "DeviceGray" | "CalGray" => ColorSpaceKind::DeviceGray,
        "DeviceRGB" | "CalRGB" => ColorSpaceKind::DeviceRgb,
        "DeviceCMYK" => ColorSpaceKind::DeviceCmyk,
        "ICCBased" => {
            let Some(profile) = arr.get(1) else {
                return ColorSpaceKind::Unknown;
            };
            if let Object::Stream { dict, .. } = doc.resolve(profile) {
                match dict.get("N").and_then(|n| n.as_i64()) {
                    Some(1) => ColorSpaceKind::DeviceGray,
                    Some(3) => ColorSpaceKind::DeviceRgb,
                    Some(4) => ColorSpaceKind::DeviceCmyk,
                    _ => ColorSpaceKind::Unknown,
                }
            } else {
                ColorSpaceKind::Unknown
            }
        }
        _ => ColorSpaceKind::Unknown,
    }
}

fn resolve_non_stroking_color_space_kind(
    doc: &PdfDoc,
    resources_dict: Option<&HashMap<String, Object>>,
    name: &str,
) -> ColorSpaceKind {
    if let Some(kind) = color_space_kind_from_name(name) {
        return kind;
    }
    let Some(color_spaces) = resources_dict
        .and_then(|d| d.get("ColorSpace"))
        .and_then(|v| doc.resolve(v).as_dict())
    else {
        return ColorSpaceKind::Unknown;
    };
    let Some(space_obj) = color_spaces.get(name) else {
        return ColorSpaceKind::Unknown;
    };
    color_space_kind_from_object(doc, space_obj)
}

fn rgb_from_cmyk(c: f64, m: f64, y: f64, k: f64) -> (f64, f64, f64) {
    let c = clamp_unit_interval(c);
    let m = clamp_unit_interval(m);
    let y = clamp_unit_interval(y);
    let k = clamp_unit_interval(k);
    (
        1.0 - (c + k).min(1.0),
        1.0 - (m + k).min(1.0),
        1.0 - (y + k).min(1.0),
    )
}

fn parse_non_stroking_rgb(
    op: &str,
    operands: &[Object],
    fill_color_space: ColorSpaceKind,
) -> Option<(f64, f64, f64)> {
    match op {
        "g" if operands.len() == 1 => {
            let g = clamp_unit_interval(operand_as_number(operands, 0)?);
            Some((g, g, g))
        }
        "rg" if operands.len() == 3 => Some((
            clamp_unit_interval(operand_as_number(operands, 0)?),
            clamp_unit_interval(operand_as_number(operands, 1)?),
            clamp_unit_interval(operand_as_number(operands, 2)?),
        )),
        "k" if operands.len() == 4 => Some(rgb_from_cmyk(
            operand_as_number(operands, 0)?,
            operand_as_number(operands, 1)?,
            operand_as_number(operands, 2)?,
            operand_as_number(operands, 3)?,
        )),
        "sc" | "scn" => match fill_color_space {
            ColorSpaceKind::DeviceGray if operands.len() == 1 => {
                let g = clamp_unit_interval(operand_as_number(operands, 0)?);
                Some((g, g, g))
            }
            ColorSpaceKind::DeviceRgb if operands.len() == 3 => Some((
                clamp_unit_interval(operand_as_number(operands, 0)?),
                clamp_unit_interval(operand_as_number(operands, 1)?),
                clamp_unit_interval(operand_as_number(operands, 2)?),
            )),
            ColorSpaceKind::DeviceCmyk if operands.len() == 4 => Some(rgb_from_cmyk(
                operand_as_number(operands, 0)?,
                operand_as_number(operands, 1)?,
                operand_as_number(operands, 2)?,
                operand_as_number(operands, 3)?,
            )),
            _ => None,
        },
        _ => None,
    }
}

fn capture_painted_fill(
    path_bbox: Option<Rect>,
    path_points: &[(f64, f64)],
    current_clip: Option<Rect>,
    text_state: &TextState,
    painted_fills: &mut Vec<PaintedFill>,
) {
    if painted_fills.len() >= MAX_TRACKED_PAINTED_FILLS {
        return;
    }
    let alpha = text_state.fill_alpha.unwrap_or(1.0);
    if alpha <= 0.0 {
        return;
    }
    let Some(mut bbox) = path_bbox.or_else(|| Rect::from_points(path_points)) else {
        return;
    };
    if let Some(clip) = current_clip {
        bbox = bbox.intersect(clip);
    }
    if bbox.is_empty() {
        return;
    }
    let bg_luminance = text_state.fill_rgb.map(|(r, g, b)| {
        let r = clamp_unit_interval(r);
        let g = clamp_unit_interval(g);
        let b = clamp_unit_interval(b);
        fill_luminance((
            r * alpha + (1.0 - alpha),
            g * alpha + (1.0 - alpha),
            b * alpha + (1.0 - alpha),
        ))
    });
    painted_fills.push(PaintedFill { bbox, bg_luminance });
}

fn text_render_mode_has_fill(render_mode: i64) -> bool {
    matches!(render_mode.rem_euclid(8), 0 | 2 | 4 | 6)
}

fn text_render_mode_has_stroke(render_mode: i64) -> bool {
    matches!(render_mode.rem_euclid(8), 1 | 2 | 5 | 6)
}

fn text_has_paint(text_state: &TextState) -> bool {
    let mode = text_state.render_mode.rem_euclid(8);
    if mode == 3 || mode == 7 {
        return false;
    }
    let fill_paints =
        text_render_mode_has_fill(mode) && !text_state.fill_alpha.is_some_and(|alpha| alpha <= 0.0);
    let stroke_paints = text_render_mode_has_stroke(mode)
        && !text_state.stroke_alpha.is_some_and(|alpha| alpha <= 0.0);
    fill_paints || stroke_paints
}

fn srgb_to_linear(channel: f64) -> f64 {
    let c = clamp_unit_interval(channel);
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

fn fill_luminance((r, g, b): (f64, f64, f64)) -> f64 {
    let rl = srgb_to_linear(r);
    let gl = srgb_to_linear(g);
    let bl = srgb_to_linear(b);
    0.2126 * rl + 0.7152 * gl + 0.0722 * bl
}

fn contrast_ratio_from_luminance(l_text: f64, l_bg: f64) -> f64 {
    let (hi, lo) = if l_text >= l_bg {
        (l_text, l_bg)
    } else {
        (l_bg, l_text)
    };
    (hi + 0.05) / (lo + 0.05)
}

fn text_contrast_against_white(text_rgb: (f64, f64, f64)) -> f64 {
    contrast_ratio_from_luminance(fill_luminance(text_rgb), 1.0)
}

fn text_is_visible_at_bbox(
    text_state: &TextState,
    bbox: Rect,
    painted_fills: &[PaintedFill],
) -> bool {
    if !text_has_paint(text_state) {
        return false;
    }
    let mode = text_state.render_mode.rem_euclid(8);
    let fill_paints =
        text_render_mode_has_fill(mode) && !text_state.fill_alpha.is_some_and(|alpha| alpha <= 0.0);
    if fill_paints {
        if let Some(text_rgb) = text_state.fill_rgb {
            let text_luminance = fill_luminance(text_rgb);
            if text_contrast_against_white(text_rgb) >= MIN_TEXT_BACKGROUND_CONTRAST_RATIO {
                return true;
            }
            for fill in painted_fills {
                if !fill.bbox.intersects(&bbox) {
                    continue;
                }
                let Some(bg_luminance) = fill.bg_luminance else {
                    return true;
                };
                let contrast = contrast_ratio_from_luminance(text_luminance, bg_luminance);
                if contrast >= MIN_TEXT_BACKGROUND_CONTRAST_RATIO {
                    return true;
                }
            }
            return false;
        }
    }
    true
}

fn text_advance_distance(text_state: &TextState, glyph_w: f64, apply_word_spacing: bool) -> f64 {
    let mut advance = glyph_w * text_state.font_size;
    if apply_word_spacing {
        advance += text_state.word_spacing;
    }
    advance += text_state.char_spacing;
    advance *= text_state.horiz_scaling / 100.0;
    advance
}

fn advance_text_matrix_by(text_state: &mut TextState, advance: f64) {
    text_state.text_matrix.e += text_state.text_matrix.a * advance;
    text_state.text_matrix.f += text_state.text_matrix.b * advance;
}

fn advance_matrix_by(matrix: &mut Matrix, advance: f64) {
    matrix.e += matrix.a * advance;
    matrix.f += matrix.b * advance;
}

fn glyph_rect_with_text_matrix(
    font: &FontMetrics,
    text_matrix: Matrix,
    ctm: &Matrix,
    font_size: f64,
    horiz_scaling: f64,
    rise: f64,
    glyph_w: f64,
) -> Rect {
    let units = font.units_to_text;
    let ascent = font.ascent * units;
    let descent = font.descent * units;
    let trm = text_matrix.multiply(Matrix {
        a: font_size * (horiz_scaling / 100.0),
        b: 0.0,
        c: 0.0,
        d: font_size,
        e: 0.0,
        f: rise,
    });
    // Column-vector convention: CTM must be left-multiplied so its translation
    // is applied in user space and not scaled by text/font matrices.
    let trm = ctm.multiply(trm);
    let (x0, y0) = trm.apply_to_point(0.0, descent);
    let (x1, y1) = trm.apply_to_point(glyph_w, ascent);
    let (x2, y2) = trm.apply_to_point(0.0, ascent);
    let (x3, y3) = trm.apply_to_point(glyph_w, descent);

    Rect {
        min_x: x0.min(x1).min(x2).min(x3),
        max_x: x0.max(x1).max(x2).max(x3),
        min_y: y0.min(y1).min(y2).min(y3),
        max_y: y0.max(y1).max(y2).max(y3),
    }
}

fn glyph_rect(font: &FontMetrics, text_state: &TextState, ctm: &Matrix, glyph_w: f64) -> Rect {
    glyph_rect_with_text_matrix(
        font,
        text_state.text_matrix,
        ctm,
        text_state.font_size,
        text_state.horiz_scaling,
        text_state.rise,
        glyph_w,
    )
}

fn text_char_count_and_has_space(text: &str) -> (usize, bool) {
    let mut char_count = 0usize;
    let mut has_space = false;
    for ch in text.chars() {
        char_count += 1;
        if ch == ' ' {
            has_space = true;
        }
    }
    (char_count, has_space)
}

fn show_text_impl<C, F>(
    page_index: usize,
    fonts: &HashMap<String, FontMetrics>,
    text_state: &mut TextState,
    ctm: &Matrix,
    clip: Option<Rect>,
    painted_fills: &[PaintedFill],
    bytes: &[u8],
    out: &mut C,
    mut push: F,
) where
    F: FnMut(CharBBox, &mut C),
{
    let font_name = match &text_state.font {
        Some(v) => v,
        None => return,
    };
    let font = match fonts.get(font_name) {
        Some(v) => v,
        None => return,
    };
    if !text_has_paint(text_state) {
        return;
    }
    let mode = text_state.render_mode.rem_euclid(8);
    let fill_paints =
        text_render_mode_has_fill(mode) && !text_state.fill_alpha.is_some_and(|alpha| alpha <= 0.0);
    let white_contrast_is_sufficient = fill_paints
        && text_state.fill_rgb.is_some_and(|text_rgb| {
            text_contrast_against_white(text_rgb) >= MIN_TEXT_BACKGROUND_CONTRAST_RATIO
        });
    let needs_visibility_check = fill_paints
        && text_state.fill_rgb.is_some()
        && !painted_fills.is_empty()
        && !white_contrast_is_sufficient;
    let font_size = text_state.font_size;
    let horiz_scaling = text_state.horiz_scaling;
    let rise = text_state.rise;

    for_each_decoded_glyph(font, bytes, |code, decoded_text| {
        let is_cid_font = font.encoding.is_some();
        let width_raw = if code <= 0xFF {
            let w = font.width(code as u8);
            if w == 0.0 && is_cid_font {
                font.width_cid(code)
            } else {
                w
            }
        } else {
            font.width_cid(code)
        };
        let glyph_w = width_raw * font.units_to_text;
        let bbox = glyph_rect(font, text_state, ctm, glyph_w);

        let mut chars = decoded_text.chars();
        let first_char = chars.next();
        let is_single_char = first_char.is_some() && chars.next().is_none();
        if is_single_char {
            let Some(ch) = first_char else {
                return;
            };
            let total_advance = text_advance_distance(text_state, glyph_w, ch == ' ');

            if let Some(c) = clip {
                if !c.intersects(&bbox) {
                    advance_text_matrix_by(text_state, total_advance);
                    return;
                }
            }
            if !needs_visibility_check || text_is_visible_at_bbox(text_state, bbox, painted_fills) {
                push(
                    CharBBox {
                        page_index,
                        ch,
                        bbox: Rectangle {
                            left: bbox.min_x,
                            top: bbox.min_y,
                            right: bbox.max_x,
                            bottom: bbox.max_y,
                        },
                    },
                    out,
                );
            }
            advance_text_matrix_by(text_state, total_advance);
            return;
        }

        let (char_count, apply_word_spacing) = text_char_count_and_has_space(decoded_text);
        let total_advance = text_advance_distance(text_state, glyph_w, apply_word_spacing);

        if let Some(c) = clip {
            if !c.intersects(&bbox) {
                advance_text_matrix_by(text_state, total_advance);
                return;
            }
        }

        if char_count == 0 {
            advance_text_matrix_by(text_state, total_advance);
            return;
        }

        let per_char_glyph_w = glyph_w / char_count as f64;
        let per_char_advance = total_advance / char_count as f64;
        let mut segment_text_matrix = text_state.text_matrix;
        for ch in decoded_text.chars() {
            let segment_bbox = glyph_rect_with_text_matrix(
                font,
                segment_text_matrix,
                ctm,
                font_size,
                horiz_scaling,
                rise,
                per_char_glyph_w,
            );
            if !needs_visibility_check
                || text_is_visible_at_bbox(text_state, segment_bbox, painted_fills)
            {
                push(
                    CharBBox {
                        page_index,
                        ch,
                        bbox: Rectangle {
                            left: segment_bbox.min_x,
                            top: segment_bbox.min_y,
                            right: segment_bbox.max_x,
                            bottom: segment_bbox.max_y,
                        },
                    },
                    out,
                );
            }
            advance_matrix_by(&mut segment_text_matrix, per_char_advance);
        }
        text_state.text_matrix = segment_text_matrix;
    });
}

fn show_text(
    page_index: usize,
    fonts: &HashMap<String, FontMetrics>,
    text_state: &mut TextState,
    ctm: &Matrix,
    clip: Option<Rect>,
    painted_fills: &[PaintedFill],
    bytes: &[u8],
    out: &mut Vec<CharBBox>,
) {
    show_text_impl(
        page_index,
        fonts,
        text_state,
        ctm,
        clip,
        painted_fills,
        bytes,
        out,
        |char_box, out| {
            out.push(char_box);
        },
    );
}

fn show_text_with_style(
    page_index: usize,
    fonts: &HashMap<String, FontMetrics>,
    text_state: &mut TextState,
    ctm: &Matrix,
    clip: Option<Rect>,
    painted_fills: &[PaintedFill],
    bytes: &[u8],
    out: &mut Vec<(CharBBox, TextStyle)>,
) {
    let style = TextStyle {
        font_size: text_state.font_size,
        fill_rgb: text_state.fill_rgb,
        font_name: text_state.font.clone(),
    };
    show_text_impl(
        page_index,
        fonts,
        text_state,
        ctm,
        clip,
        painted_fills,
        bytes,
        out,
        |char_box, out| out.push((char_box, style.clone())),
    );
}

fn show_text_for_block_accumulator(
    page_index: usize,
    fonts: &HashMap<String, FontMetrics>,
    text_state: &mut TextState,
    ctm: &Matrix,
    clip: Option<Rect>,
    painted_fills: &[PaintedFill],
    bytes: &[u8],
    out: &mut BlockAccumulator,
) {
    let font_size = text_state.font_size;
    show_text_impl(
        page_index,
        fonts,
        text_state,
        ctm,
        clip,
        painted_fills,
        bytes,
        out,
        |char_box, out| {
            if char_box.ch.is_whitespace() {
                out.push_whitespace();
            } else {
                out.push_char(char_box, font_size);
            }
        },
    );
}

fn collect_pages(doc: &PdfDoc) -> Vec<Object> {
    let mut pages = Vec::new();
    let root = doc
        .trailer
        .as_ref()
        .and_then(|t| t.as_dict())
        .and_then(|d| d.get("Root"))
        .map(|v| doc.resolve(v).clone());
    if let Some(root) = root {
        let pages_ref = root.as_dict().and_then(|d| d.get("Pages"));
        if let Some(pages_root) = pages_ref {
            let resolved = doc.resolve(pages_root).clone();
            walk_page_tree(doc, &resolved, None, None, None, &mut pages);
        }
    }
    pages
}

fn page_layouts_for_pages(doc: &PdfDoc, pages: &[Object]) -> Vec<PageLayout> {
    let mut layouts = Vec::with_capacity(pages.len());
    let mut y_offset = 0.0;
    for page in pages {
        let page_dict = page.as_dict();
        let media_box = page_dict
            .and_then(|d| d.get("MediaBox").or_else(|| d.get("CropBox")))
            .map(|v| doc.resolve(v))
            .and_then(parse_rect)
            .unwrap_or((0.0, 0.0, 0.0, 0.0));
        let (x_min, y_min, x_max, y_max) = normalize_rect_tuple(media_box);
        let width = (x_max - x_min).max(0.0);
        let height = (y_max - y_min).max(0.0);
        let rotation = normalized_page_rotation(
            page_dict
                .and_then(|d| d.get("Rotate"))
                .map(|v| doc.resolve(v)),
        );
        let (rotated_width, rotated_height) = if rotation == 90 || rotation == 270 {
            (height, width)
        } else {
            (width, height)
        };
        layouts.push(PageLayout {
            x_min,
            y_min,
            width,
            height,
            rotation,
            rotated_width,
            rotated_height,
            y_offset,
        });
        y_offset += rotated_height;
    }
    layouts
}

fn canonicalize_char_bboxes(chars: Vec<CharBBox>, layouts: &[PageLayout]) -> Vec<CharBBox> {
    chars
        .into_iter()
        .map(|ch| CharBBox {
            page_index: ch.page_index,
            ch: ch.ch,
            bbox: canonicalize_rect(ch.bbox, ch.page_index, layouts),
        })
        .collect()
}

fn canonicalize_char_bboxes_with_style(
    chars: Vec<(CharBBox, TextStyle)>,
    layouts: &[PageLayout],
) -> Vec<(CharBBox, TextStyle)> {
    chars
        .into_iter()
        .map(|(ch, style)| {
            (
                CharBBox {
                    page_index: ch.page_index,
                    ch: ch.ch,
                    bbox: canonicalize_rect(ch.bbox, ch.page_index, layouts),
                },
                style,
            )
        })
        .collect()
}

fn canonicalize_text_blocks(blocks: Vec<TextBlock>, layouts: &[PageLayout]) -> Vec<TextBlock> {
    blocks
        .into_iter()
        .map(|block| TextBlock {
            page_index: block.page_index,
            text: block.text,
            bbox: canonicalize_rect(block.bbox, block.page_index, layouts),
        })
        .collect()
}

fn canonicalize_rect(rect: Rectangle, page_index: usize, layouts: &[PageLayout]) -> Rectangle {
    let normalized = normalize_rectangle(rect);
    let Some(layout) = layouts.get(page_index).copied() else {
        return normalized;
    };
    debug_assert!(layout.width >= 0.0);
    debug_assert!(layout.height >= 0.0);
    debug_assert!(layout.rotated_width >= 0.0);
    debug_assert!(layout.rotated_height >= 0.0);
    let local_points = [
        (
            normalized.left - layout.x_min,
            normalized.top - layout.y_min,
        ),
        (
            normalized.left - layout.x_min,
            normalized.bottom - layout.y_min,
        ),
        (
            normalized.right - layout.x_min,
            normalized.top - layout.y_min,
        ),
        (
            normalized.right - layout.x_min,
            normalized.bottom - layout.y_min,
        ),
    ];
    let rotated_points = local_points.map(|(x, y)| rotate_page_point(layout, x, y));
    let (min_x, max_x, min_y, max_y) = rotated_points.into_iter().fold(
        (
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::INFINITY,
            f64::NEG_INFINITY,
        ),
        |(min_x, max_x, min_y, max_y), (x, y)| {
            (min_x.min(x), max_x.max(x), min_y.min(y), max_y.max(y))
        },
    );
    normalize_rectangle(Rectangle {
        left: min_x,
        right: max_x,
        top: layout.y_offset + (layout.rotated_height - max_y),
        bottom: layout.y_offset + (layout.rotated_height - min_y),
    })
}

fn normalize_rectangle(rect: Rectangle) -> Rectangle {
    Rectangle {
        left: rect.left.min(rect.right),
        right: rect.left.max(rect.right),
        top: rect.top.min(rect.bottom),
        bottom: rect.top.max(rect.bottom),
    }
}

fn walk_page_tree(
    doc: &PdfDoc,
    node: &Object,
    inherited_resources: Option<Object>,
    inherited_media_box: Option<Object>,
    inherited_rotate: Option<Object>,
    out: &mut Vec<Object>,
) {
    let node = doc.resolve(node).clone();
    let dict = match node.as_dict() {
        Some(v) => v,
        None => return,
    };
    let resources = dict
        .get("Resources")
        .map(|r| doc.resolve(r).clone())
        .or(inherited_resources);
    let media_box = dict
        .get("MediaBox")
        .map(|b| doc.resolve(b).clone())
        .or(inherited_media_box);
    let rotate = dict
        .get("Rotate")
        .map(|r| doc.resolve(r).clone())
        .or(inherited_rotate);
    match dict.get("Type").and_then(|v| v.as_name()) {
        Some("Page") => {
            let mut page = node.clone();
            if let Object::Dictionary(ref mut page_dict) = page {
                if resources.is_some() && !page_dict.contains_key("Resources") {
                    page_dict.insert("Resources".to_string(), resources.clone().unwrap());
                }
                if media_box.is_some() && !page_dict.contains_key("MediaBox") {
                    page_dict.insert("MediaBox".to_string(), media_box.clone().unwrap());
                }
                if rotate.is_some() && !page_dict.contains_key("Rotate") {
                    page_dict.insert("Rotate".to_string(), rotate.clone().unwrap());
                }
                let normalized_rotate =
                    normalized_page_rotation(page_dict.get("Rotate").or_else(|| rotate.as_ref()));
                page_dict.insert(
                    "Rotate".to_string(),
                    Object::Integer(normalized_rotate as i64),
                );
            }
            out.push(page);
        }
        _ => {
            if let Some(Object::Array(kids)) = dict.get("Kids") {
                for kid in kids {
                    walk_page_tree(
                        doc,
                        kid,
                        resources.clone(),
                        media_box.clone(),
                        rotate.clone(),
                        out,
                    );
                }
            }
        }
    }
}

fn normalized_page_rotation(rotation: Option<&Object>) -> i32 {
    let raw = rotation
        .and_then(|v| v.as_i64().or_else(|| v.as_f64().map(|n| n.round() as i64)))
        .unwrap_or(0);
    match raw.rem_euclid(360) {
        0 => 0,
        90 => 90,
        180 => 180,
        270 => 270,
        _ => 0,
    }
}

fn rotate_page_point(layout: PageLayout, x: f64, y: f64) -> (f64, f64) {
    match layout.rotation {
        90 => (y, layout.width - x),
        180 => (layout.width - x, layout.height - y),
        270 => (layout.height - y, x),
        _ => (x, y),
    }
}

fn page_resources_and_contents<'a>(
    doc: &'a PdfDoc,
    page: &'a Object,
) -> (Option<&'a Object>, Vec<&'a Object>) {
    let dict = match page.as_dict() {
        Some(v) => v,
        None => return (None, Vec::new()),
    };
    let resources = dict.get("Resources").map(|r| doc.resolve(r));
    let mut contents = Vec::new();
    if let Some(content) = dict.get("Contents") {
        let content = doc.resolve(content);
        match content {
            Object::Array(items) => {
                for item in items {
                    contents.push(doc.resolve(item));
                }
            }
            _ => contents.push(content),
        }
    }
    (resources, contents)
}

fn annotation_appearance_streams<'a>(
    doc: &'a PdfDoc,
    page: &'a Object,
) -> Vec<(Vec<u8>, Option<&'a Object>, Matrix)> {
    let mut out = Vec::new();
    let dict = match page.as_dict() {
        Some(v) => v,
        None => return out,
    };
    let annots = match dict.get("Annots").and_then(|v| doc.resolve(v).as_array()) {
        Some(v) => v,
        None => return out,
    };
    for annot in annots {
        let annot = doc.resolve(annot);
        let adict = match annot.as_dict() {
            Some(v) => v,
            None => continue,
        };
        let rect = adict.get("Rect").and_then(parse_rect);
        let ap = adict.get("AP").and_then(|v| doc.resolve(v).as_dict());
        let n = match ap.and_then(|d| d.get("N")) {
            Some(v) => v,
            None => continue,
        };
        let stream = match resolve_appearance_stream(doc, n) {
            Some(v) => v,
            None => continue,
        };
        let (dict, bytes) = match &stream {
            Object::Stream { dict, .. } => {
                let bytes = match decode_stream(doc, &stream) {
                    Some(v) => v,
                    None => continue,
                };
                (dict, bytes)
            }
            _ => continue,
        };
        let resources = dict.get("Resources").map(|r| doc.resolve(r));
        let bbox = dict.get("BBox").and_then(parse_rect);
        let stream_matrix = dict
            .get("Matrix")
            .and_then(|v| doc.resolve(v).as_array())
            .and_then(matrix_from_array)
            .unwrap_or_else(Matrix::identity);
        let rect_map = match (rect, bbox) {
            (Some(r), Some(b)) => map_bbox_to_rect(b, r),
            (Some(r), None) => map_bbox_to_rect(r, r),
            _ => Matrix::identity(),
        };
        let matrix = rect_map.multiply(stream_matrix);
        out.push((bytes, resources, matrix));
    }
    out
}

fn resolve_appearance_stream<'a>(doc: &'a PdfDoc, obj: &'a Object) -> Option<&'a Object> {
    let resolved = doc.resolve(obj);
    match resolved {
        Object::Stream { .. } => Some(resolved),
        Object::Dictionary(dict) => {
            for value in dict.values() {
                if let Some(stream) = resolve_appearance_stream(doc, value) {
                    return Some(stream);
                }
            }
            None
        }
        _ => None,
    }
}

fn parse_rect(obj: &Object) -> Option<(f64, f64, f64, f64)> {
    let arr = obj.as_array()?;
    if arr.len() < 4 {
        return None;
    }
    let x0 = arr[0].as_f64()?;
    let y0 = arr[1].as_f64()?;
    let x1 = arr[2].as_f64()?;
    let y1 = arr[3].as_f64()?;
    Some(normalize_rect_tuple((x0, y0, x1, y1)))
}

fn matrix_from_array(arr: &[Object]) -> Option<Matrix> {
    if arr.len() != 6 {
        return None;
    }
    Some(Matrix {
        a: arr[0].as_f64().unwrap_or(1.0),
        b: arr[1].as_f64().unwrap_or(0.0),
        c: arr[2].as_f64().unwrap_or(0.0),
        d: arr[3].as_f64().unwrap_or(1.0),
        e: arr[4].as_f64().unwrap_or(0.0),
        f: arr[5].as_f64().unwrap_or(0.0),
    })
}

fn map_bbox_to_rect(bbox: (f64, f64, f64, f64), rect: (f64, f64, f64, f64)) -> Matrix {
    let (bx0, by0, bx1, by1) = normalize_rect_tuple(bbox);
    let (rx0, ry0, rx1, ry1) = normalize_rect_tuple(rect);
    let bw = bx1 - bx0;
    let bh = by1 - by0;
    let rw = rx1 - rx0;
    let rh = ry1 - ry0;
    let scale_x = if bw != 0.0 { rw / bw } else { 1.0 };
    let scale_y = if bh != 0.0 { rh / bh } else { 1.0 };
    let tx = rx0 - bx0 * scale_x;
    let ty = ry0 - by0 * scale_y;
    Matrix {
        a: scale_x,
        b: 0.0,
        c: 0.0,
        d: scale_y,
        e: tx,
        f: ty,
    }
}

fn normalize_rect_tuple(r: (f64, f64, f64, f64)) -> (f64, f64, f64, f64) {
    let (x0, y0, x1, y1) = r;
    (x0.min(x1), y0.min(y1), x0.max(x1), y0.max(y1))
}

fn build_font_map(doc: &PdfDoc, resources: Option<&Object>) -> HashMap<String, FontMetrics> {
    let mut out = HashMap::new();
    let resources = match resources.and_then(|r| r.as_dict()) {
        Some(v) => v,
        None => return out,
    };
    let font_dict = match resources.get("Font").and_then(|f| doc.resolve(f).as_dict()) {
        Some(v) => v,
        None => return out,
    };
    for (name, font_obj) in font_dict {
        let font_obj = doc.resolve(font_obj);
        let dict = match font_obj.as_dict() {
            Some(v) => v,
            None => continue,
        };
        let base_font = dict
            .get("BaseFont")
            .and_then(|v| v.as_name())
            .map(|v| v.to_string());
        let first_char = dict.get("FirstChar").and_then(|v| v.as_i64()).unwrap_or(0);
        let widths = dict
            .get("Widths")
            .and_then(|v| doc.resolve(v).as_array())
            .map(|arr| {
                arr.iter()
                    .map(|o| o.as_f64().unwrap_or(0.0))
                    .collect::<Vec<f64>>()
            })
            .unwrap_or_default();
        let (font_matrix_x, font_matrix_y) = dict
            .get("FontMatrix")
            .and_then(|v| doc.resolve(v).as_array())
            .map(|arr| {
                let x = arr.first().and_then(|v| v.as_f64()).unwrap_or(0.001);
                let y = arr.get(3).and_then(|v| v.as_f64()).unwrap_or(x);
                (x, y)
            })
            .unwrap_or((0.001, 0.001));
        let descriptor = dict
            .get("FontDescriptor")
            .and_then(|v| doc.resolve(v).as_dict());
        let (mut ascent, mut descent) = descriptor
            .map(|d| {
                (
                    d.get("Ascent").and_then(|v| v.as_f64()).unwrap_or(800.0),
                    d.get("Descent").and_then(|v| v.as_f64()).unwrap_or(-200.0),
                )
            })
            .unwrap_or((800.0, -200.0));

        let mut units_to_text = font_matrix_x;
        let mut widths_table = widths;
        let mut cid_widths: Vec<CidWidthRange> = Vec::new();
        let mut cid_default_width: i64 = 1000;
        let mut single_byte_encoding = SingleByteEncoding::WinAnsi;
        let mut single_byte_overrides = HashMap::new();
        let to_unicode = dict
            .get("ToUnicode")
            .and_then(|v| decode_stream(doc, v))
            .map(|data| parse_cmap(&data));

        let subtype = dict.get("Subtype").and_then(|v| v.as_name());
        if subtype == Some("Type3")
            && descriptor.is_none()
            && let Some((_, bbox_min_y, _, bbox_max_y)) = dict
                .get("FontBBox")
                .map(|v| doc.resolve(v))
                .and_then(parse_rect)
        {
            let font_matrix_y_for_metrics = if font_matrix_y.abs() > TYPE3_UNITS_EPSILON {
                font_matrix_y
            } else {
                units_to_text
            };
            let bbox_text_height =
                (bbox_max_y - bbox_min_y).abs() * font_matrix_y_for_metrics.abs();
            if (TYPE3_TEXT_HEIGHT_MIN..=TYPE3_TEXT_HEIGHT_MAX).contains(&bbox_text_height) {
                ascent = bbox_max_y;
                descent = bbox_min_y;
            } else {
                let midpoint = (bbox_max_y + bbox_min_y) * 0.5;
                let safe_units = units_to_text.abs().max(TYPE3_UNITS_EPSILON);
                let fallback_half_height = (TYPE3_FALLBACK_TEXT_HEIGHT * 0.5) / safe_units;
                ascent = midpoint + fallback_half_height;
                descent = midpoint - fallback_half_height;
            }
        }
        let mut descendant_dict: Option<&HashMap<String, Object>> = None;
        let mut encoding: Option<EncodingMap> = None;
        if subtype == Some("Type0") {
            if let Some(Object::Array(desc)) = dict.get("DescendantFonts") {
                if let Some(first) = desc.get(0) {
                    descendant_dict = doc.resolve(first).as_dict();
                }
            }
            if let Some(enc) = dict.get("Encoding") {
                encoding = parse_encoding(doc, enc);
            }
        } else if let Some(enc) = dict.get("Encoding") {
            if let Some((parsed, overrides)) = parse_single_byte_encoding(doc, enc) {
                single_byte_encoding = parsed;
                single_byte_overrides = overrides;
            }
        }

        if let Some(desc) = descendant_dict {
            if let Some(dw) = desc
                .get("DW")
                .map(|v| doc.resolve(v))
                .and_then(|v| v.as_f64())
            {
                cid_default_width = dw.round() as i64;
            }
            if let Some(w) = desc
                .get("W")
                .map(|v| doc.resolve(v))
                .and_then(|v| v.as_array())
            {
                let resolved_widths = resolve_cid_width_array(doc, w);
                cid_widths = parse_cid_widths(&resolved_widths);
            }
        }

        let metrics_source = descendant_dict.unwrap_or(dict);
        let canonical_base14 = base_font.as_deref().and_then(canonical_base14_spec);

        let has_cid_widths = !cid_widths.is_empty();
        if has_cid_widths {
            // CIDFont /W and /DW are in 1/1000 text space units.
            units_to_text = 0.001;
        }

        let needs_ttf_fallback = widths_table.is_empty() && !has_cid_widths;
        let mut ttf_metrics: Option<TtfMetrics> = None;
        if needs_ttf_fallback {
            ttf_metrics = extract_ttf_metrics(doc, metrics_source);
            if ttf_metrics.is_none() && canonical_base14.is_none() {
                ttf_metrics = base_font.as_deref().and_then(extract_system_ttf_metrics);
            }
        }
        if let Some(ttf_metrics) = ttf_metrics.as_ref() {
            widths_table = ttf_metrics.widths.clone();
            ascent = ttf_metrics.ascent;
            descent = ttf_metrics.descent;
            units_to_text = ttf_metrics.units_to_text;
        }

        let is_non_embedded = is_non_embedded_font(metrics_source, doc);
        if is_non_embedded
            && canonical_base14.is_some_and(|spec| {
                matches!(
                    spec.family,
                    Base14Family::Courier
                        | Base14Family::Times
                        | Base14Family::Symbol
                        | Base14Family::ZapfDingbats
                )
            })
            && let Some(base14_metrics) =
                canonical_base14.and_then(|_| base_font.as_deref().and_then(base14_metrics))
        {
            // Non-embedded base-14-compatible fonts can report unstable descriptor metrics.
            // Prefer deterministic base-14 vertical metrics and fallback widths when needed.
            ascent = base14_metrics.ascent;
            descent = base14_metrics.descent;
            units_to_text = base14_metrics.units_to_text;
            if is_unusable_widths(&widths_table) && !has_cid_widths {
                widths_table = base14_metrics.widths;
            }
            if cid_default_width <= 0 {
                cid_default_width = base14_metrics.cid_default_width;
            }
        }

        if widths_table.is_empty()
            && !has_cid_widths
            && let Some(base14_metrics) = base_font.as_deref().and_then(base14_metrics)
        {
            // Preserve ToUnicode/encoding maps; replacing the entire struct here can lose CID
            // decoding for Type0 fonts and produce garbled byte-level fallback text.
            widths_table = base14_metrics.widths;
        }

        let metrics = FontMetrics {
            first_char,
            widths: widths_table,
            ascent,
            descent,
            units_to_text,
            to_unicode: to_unicode.clone(),
            encoding: encoding.clone(),
            single_byte_encoding,
            single_byte_overrides,
            cid_widths: cid_widths.clone(),
            cid_default_width,
        };
        out.insert(name.clone(), metrics);
    }
    out
}

fn decode_contents(doc: &PdfDoc, contents: &[&Object]) -> Vec<u8> {
    let mut out = Vec::new();
    for content in contents {
        if let Some(decoded) = decode_stream(doc, content) {
            out.extend_from_slice(&decoded);
            out.push(b'\n');
        }
    }
    out
}

fn flate_decode(data: &[u8]) -> Vec<u8> {
    use std::io::Read;
    let mut decoder = flate2::read::ZlibDecoder::new(data);
    let mut out = Vec::new();
    let _ = decoder.read_to_end(&mut out);
    out
}

#[derive(Clone)]
struct TtfMetrics {
    widths: Vec<f64>,
    ascent: f64,
    descent: f64,
    units_to_text: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Base14Family {
    Helvetica,
    Courier,
    Times,
    Symbol,
    ZapfDingbats,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Base14Style {
    Regular,
    Bold,
    Italic,
    BoldItalic,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Base14Spec {
    family: Base14Family,
    style: Base14Style,
}

fn extract_ttf_metrics(doc: &PdfDoc, dict: &HashMap<String, Object>) -> Option<TtfMetrics> {
    let descriptor = dict
        .get("FontDescriptor")
        .and_then(|v| doc.resolve(v).as_dict())?;
    let font_file = descriptor.get("FontFile2").map(|v| doc.resolve(v))?;
    let data = decode_stream(doc, font_file)?;
    let face = ttf_parser::Face::parse(&data, 0).ok()?;
    build_ttf_metrics_from_face(&face)
}

fn build_ttf_metrics_from_face(face: &ttf_parser::Face<'_>) -> Option<TtfMetrics> {
    let units_per_em = face.units_per_em() as f64;
    if units_per_em == 0.0 {
        return None;
    }

    let mut widths = vec![0.0; 256];
    for code in 0u16..=255 {
        let uni = WIN_ANSI[code as usize];
        if uni == 0 {
            continue;
        }
        if let Some(ch) = char::from_u32(uni as u32) {
            if let Some(gid) = face.glyph_index(ch) {
                if let Some(adv) = face.glyph_hor_advance(gid) {
                    widths[code as usize] = adv as f64;
                }
            }
        }
    }

    Some(TtfMetrics {
        widths,
        ascent: face.ascender() as f64,
        descent: face.descender() as f64,
        units_to_text: 1.0 / units_per_em,
    })
}

fn system_ttf_metrics_cache() -> &'static RwLock<HashMap<String, Option<TtfMetrics>>> {
    static CACHE: OnceLock<RwLock<HashMap<String, Option<TtfMetrics>>>> = OnceLock::new();
    CACHE.get_or_init(|| RwLock::new(HashMap::new()))
}

fn extract_system_ttf_metrics(base_font: &str) -> Option<TtfMetrics> {
    let base_norm = normalize_font_name(base_font);
    if let Ok(cache) = system_ttf_metrics_cache().read()
        && let Some(cached) = cache.get(&base_norm)
    {
        return cached.clone();
    }

    let candidates = system_font_candidates(base_font);
    let mut found: Option<TtfMetrics> = None;
    for path in candidates {
        let Ok(data) = fs::read(path) else {
            continue;
        };
        if let Some(count) = ttf_parser::fonts_in_collection(&data) {
            for idx in 0..count {
                if let Ok(face) = ttf_parser::Face::parse(&data, idx) {
                    if face_name_matches(&face, &base_norm) {
                        found = build_ttf_metrics_from_face(&face);
                        if found.is_some() {
                            break;
                        }
                    }
                }
            }
        } else if let Ok(face) = ttf_parser::Face::parse(&data, 0) {
            if face_name_matches(&face, &base_norm) {
                found = build_ttf_metrics_from_face(&face);
            }
        }
        if found.is_some() {
            break;
        }
    }

    if let Ok(mut cache) = system_ttf_metrics_cache().write() {
        cache.insert(base_norm, found.clone());
    }

    found
}

fn normalize_font_name(name: &str) -> String {
    strip_subset_prefix(name)
        .chars()
        .filter(|c| c.is_alphanumeric())
        .flat_map(|c| c.to_lowercase())
        .collect()
}

fn strip_subset_prefix(name: &str) -> &str {
    if let Some((prefix, rest)) = name.split_once('+')
        && prefix.len() == 6
        && prefix.chars().all(|ch| ch.is_ascii_uppercase())
    {
        return rest;
    }
    name
}

fn canonical_base14_spec(name: &str) -> Option<Base14Spec> {
    let normalized = normalize_font_name(name);
    match normalized.as_str() {
        "helvetica" => Some(Base14Spec {
            family: Base14Family::Helvetica,
            style: Base14Style::Regular,
        }),
        "helveticabold" => Some(Base14Spec {
            family: Base14Family::Helvetica,
            style: Base14Style::Bold,
        }),
        "helveticaoblique" => Some(Base14Spec {
            family: Base14Family::Helvetica,
            style: Base14Style::Italic,
        }),
        "helveticaboldoblique" => Some(Base14Spec {
            family: Base14Family::Helvetica,
            style: Base14Style::BoldItalic,
        }),
        "courier" | "couriernew" | "couriernewpsmt" => Some(Base14Spec {
            family: Base14Family::Courier,
            style: Base14Style::Regular,
        }),
        "courierbold" | "couriernewbold" | "couriernewpsboldmt" => Some(Base14Spec {
            family: Base14Family::Courier,
            style: Base14Style::Bold,
        }),
        "courieroblique" | "couriernewitalic" | "couriernewoblique" | "couriernewpsitalicmt" => {
            Some(Base14Spec {
                family: Base14Family::Courier,
                style: Base14Style::Italic,
            })
        }
        "courierboldoblique"
        | "couriernewbolditalic"
        | "couriernewboldoblique"
        | "couriernewpsbolditalicmt" => Some(Base14Spec {
            family: Base14Family::Courier,
            style: Base14Style::BoldItalic,
        }),
        "timesroman" | "timesnewromanpsmt" => Some(Base14Spec {
            family: Base14Family::Times,
            style: Base14Style::Regular,
        }),
        "timesbold" | "timesnewromanpsboldmt" => Some(Base14Spec {
            family: Base14Family::Times,
            style: Base14Style::Bold,
        }),
        "timesitalic" | "timesnewromanpsitalicmt" => Some(Base14Spec {
            family: Base14Family::Times,
            style: Base14Style::Italic,
        }),
        "timesbolditalic" | "timesnewromanpsbolditalicmt" => Some(Base14Spec {
            family: Base14Family::Times,
            style: Base14Style::BoldItalic,
        }),
        "symbol" | "symbolmt" => Some(Base14Spec {
            family: Base14Family::Symbol,
            style: Base14Style::Regular,
        }),
        "zapfdingbats" | "itczapfdingbatsstd" => Some(Base14Spec {
            family: Base14Family::ZapfDingbats,
            style: Base14Style::Regular,
        }),
        _ => None,
    }
}

fn is_unusable_widths(widths: &[f64]) -> bool {
    widths.is_empty() || widths.iter().all(|w| *w <= 0.0)
}

fn is_non_embedded_font(metrics_source: &HashMap<String, Object>, doc: &PdfDoc) -> bool {
    let Some(descriptor) = metrics_source
        .get("FontDescriptor")
        .and_then(|v| doc.resolve(v).as_dict())
    else {
        return true;
    };
    !descriptor.contains_key("FontFile")
        && !descriptor.contains_key("FontFile2")
        && !descriptor.contains_key("FontFile3")
}

fn face_name_matches(face: &ttf_parser::Face<'_>, base_norm: &str) -> bool {
    for name in face.names() {
        if name.is_unicode()
            && (name.name_id == ttf_parser::name_id::POST_SCRIPT_NAME
                || name.name_id == ttf_parser::name_id::FULL_NAME)
        {
            if let Some(value) = name.to_string() {
                if normalize_font_name(&value) == base_norm {
                    return true;
                }
            }
        }
    }
    false
}

fn system_font_candidates(base_font: &str) -> Vec<&'static str> {
    match canonical_base14_spec(base_font) {
        Some(Base14Spec {
            family: Base14Family::Helvetica,
            ..
        }) => vec![
            // macOS
            "/System/Library/Fonts/Helvetica.ttc",
            "/System/Library/Fonts/Supplemental/Helvetica.ttf",
            "/System/Library/Fonts/Supplemental/Helvetica Bold.ttf",
            "/System/Library/Fonts/Supplemental/Helvetica Oblique.ttf",
            "/System/Library/Fonts/Supplemental/Helvetica Bold Oblique.ttf",
            "/Library/Fonts/Helvetica.ttf",
            "/Library/Fonts/Helvetica Bold.ttf",
            // Linux
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Oblique.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-BoldOblique.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Italic.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-BoldItalic.ttf",
            "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
            "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation2/LiberationSans-Italic.ttf",
            "/usr/share/fonts/truetype/liberation2/LiberationSans-BoldItalic.ttf",
            "/usr/share/fonts/opentype/urw-base35/NimbusSans-Regular.otf",
            "/usr/share/fonts/opentype/urw-base35/NimbusSans-Bold.otf",
            "/usr/share/fonts/opentype/urw-base35/NimbusSans-Italic.otf",
            "/usr/share/fonts/opentype/urw-base35/NimbusSans-BoldItalic.otf",
            // Windows
            "C:\\Windows\\Fonts\\arial.ttf",
            "C:\\Windows\\Fonts\\arialbd.ttf",
            "C:\\Windows\\Fonts\\ariali.ttf",
            "C:\\Windows\\Fonts\\arialbi.ttf",
        ],
        Some(Base14Spec {
            family: Base14Family::Courier,
            ..
        }) => vec![
            // macOS
            "/System/Library/Fonts/Supplemental/Courier New.ttf",
            "/System/Library/Fonts/Supplemental/Courier New Bold.ttf",
            "/System/Library/Fonts/Supplemental/Courier New Italic.ttf",
            "/System/Library/Fonts/Supplemental/Courier New Bold Italic.ttf",
            "/Library/Fonts/Courier New.ttf",
            // Linux
            "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationMono-Bold.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationMono-Italic.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationMono-BoldItalic.ttf",
            "/usr/share/fonts/truetype/liberation2/LiberationMono-Regular.ttf",
            "/usr/share/fonts/truetype/liberation2/LiberationMono-Bold.ttf",
            "/usr/share/fonts/truetype/liberation2/LiberationMono-Italic.ttf",
            "/usr/share/fonts/truetype/liberation2/LiberationMono-BoldItalic.ttf",
            "/usr/share/fonts/opentype/urw-base35/NimbusMonoPS-Regular.otf",
            "/usr/share/fonts/opentype/urw-base35/NimbusMonoPS-Bold.otf",
            "/usr/share/fonts/opentype/urw-base35/NimbusMonoPS-Italic.otf",
            "/usr/share/fonts/opentype/urw-base35/NimbusMonoPS-BoldItalic.otf",
            // Windows
            "C:\\Windows\\Fonts\\cour.ttf",
            "C:\\Windows\\Fonts\\courbd.ttf",
            "C:\\Windows\\Fonts\\couri.ttf",
            "C:\\Windows\\Fonts\\courbi.ttf",
        ],
        Some(Base14Spec {
            family: Base14Family::Times,
            ..
        }) => vec![
            // macOS
            "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
            "/System/Library/Fonts/Supplemental/Times New Roman Bold.ttf",
            "/System/Library/Fonts/Supplemental/Times New Roman Italic.ttf",
            "/System/Library/Fonts/Supplemental/Times New Roman Bold Italic.ttf",
            "/Library/Fonts/Times New Roman.ttf",
            // Linux
            "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Italic.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSerif-BoldItalic.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSerif-Italic.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSerif-BoldItalic.ttf",
            "/usr/share/fonts/truetype/liberation2/LiberationSerif-Regular.ttf",
            "/usr/share/fonts/truetype/liberation2/LiberationSerif-Bold.ttf",
            "/usr/share/fonts/truetype/liberation2/LiberationSerif-Italic.ttf",
            "/usr/share/fonts/truetype/liberation2/LiberationSerif-BoldItalic.ttf",
            "/usr/share/fonts/opentype/urw-base35/NimbusRoman-Regular.otf",
            "/usr/share/fonts/opentype/urw-base35/NimbusRoman-Bold.otf",
            "/usr/share/fonts/opentype/urw-base35/NimbusRoman-Italic.otf",
            "/usr/share/fonts/opentype/urw-base35/NimbusRoman-BoldItalic.otf",
            // Windows
            "C:\\Windows\\Fonts\\times.ttf",
            "C:\\Windows\\Fonts\\timesbd.ttf",
            "C:\\Windows\\Fonts\\timesi.ttf",
            "C:\\Windows\\Fonts\\timesbi.ttf",
        ],
        _ => Vec::new(),
    }
}

fn decode_stream(doc: &PdfDoc, obj: &Object) -> Option<Vec<u8>> {
    match doc.resolve(obj) {
        Object::Stream { dict, data } => {
            if stream_uses_flate_filter(doc, dict) {
                return Some(flate_decode(data));
            }
            Some(data.clone())
        }
        _ => None,
    }
}

const STREAM_REF_RESOLVE_MAX_DEPTH: usize = 8;

fn stream_uses_flate_filter(doc: &PdfDoc, dict: &HashMap<String, Object>) -> bool {
    let Some(filter) = dict.get("Filter") else {
        return false;
    };
    let filter = resolve_object_reference_chain(doc, filter, STREAM_REF_RESOLVE_MAX_DEPTH);
    match filter {
        Object::Name(name) => name == "FlateDecode",
        Object::Array(items) => match items.first() {
            Some(first) => {
                let first =
                    resolve_object_reference_chain(doc, first, STREAM_REF_RESOLVE_MAX_DEPTH);
                matches!(first, Object::Name(name) if name == "FlateDecode")
            }
            None => false,
        },
        _ => false,
    }
}

fn resolve_object_reference_chain<'a>(
    doc: &'a PdfDoc,
    mut obj: &'a Object,
    max_depth: usize,
) -> &'a Object {
    for _ in 0..max_depth {
        match obj {
            Object::Reference { .. } => {
                let resolved = doc.resolve(obj);
                if std::ptr::eq(resolved, obj) {
                    break;
                }
                obj = resolved;
            }
            _ => break,
        }
    }
    obj
}

fn base14_metrics(name: &str) -> Option<FontMetrics> {
    match canonical_base14_spec(name) {
        Some(Base14Spec {
            family: Base14Family::Helvetica,
            ..
        }) => {
            let mut widths = vec![0.0; 256];
            let base = [
                (32, 278),
                (33, 278),
                (34, 355),
                (35, 556),
                (36, 556),
                (37, 889),
                (38, 667),
                (39, 191),
                (40, 333),
                (41, 333),
                (42, 389),
                (43, 584),
                (44, 278),
                (45, 333),
                (46, 278),
                (47, 278),
                (48, 556),
                (49, 556),
                (50, 556),
                (51, 556),
                (52, 556),
                (53, 556),
                (54, 556),
                (55, 556),
                (56, 556),
                (57, 556),
                (58, 278),
                (59, 278),
                (60, 584),
                (61, 584),
                (62, 584),
                (63, 556),
                (64, 1015),
                (65, 667),
                (66, 667),
                (67, 722),
                (68, 722),
                (69, 667),
                (70, 611),
                (71, 778),
                (72, 722),
                (73, 278),
                (74, 500),
                (75, 667),
                (76, 556),
                (77, 833),
                (78, 722),
                (79, 778),
                (80, 667),
                (81, 778),
                (82, 722),
                (83, 667),
                (84, 611),
                (85, 722),
                (86, 667),
                (87, 944),
                (88, 667),
                (89, 667),
                (90, 611),
                (91, 278),
                (92, 278),
                (93, 278),
                (94, 469),
                (95, 556),
                (96, 222),
                (97, 556),
                (98, 556),
                (99, 500),
                (100, 556),
                (101, 556),
                (102, 278),
                (103, 556),
                (104, 556),
                (105, 222),
                (106, 222),
                (107, 500),
                (108, 222),
                (109, 833),
                (110, 556),
                (111, 556),
                (112, 556),
                (113, 556),
                (114, 333),
                (115, 500),
                (116, 278),
                (117, 556),
                (118, 500),
                (119, 722),
                (120, 500),
                (121, 500),
                (122, 500),
                (123, 334),
                (124, 260),
                (125, 334),
                (126, 584),
            ];
            for (code, width) in base {
                widths[code as usize] = width as f64;
            }
            Some(FontMetrics {
                first_char: 0,
                widths,
                ascent: 718.0,
                descent: -207.0,
                units_to_text: 0.001,
                to_unicode: None,
                encoding: None,
                single_byte_encoding: SingleByteEncoding::WinAnsi,
                single_byte_overrides: HashMap::new(),
                cid_widths: Vec::new(),
                cid_default_width: 1000,
            })
        }
        Some(Base14Spec {
            family: Base14Family::Courier,
            ..
        }) => {
            let mut widths = vec![600.0; 256];
            // Keep control characters as zero width.
            widths.iter_mut().take(32).for_each(|w| *w = 0.0);
            Some(FontMetrics {
                first_char: 0,
                widths,
                ascent: 629.0,
                descent: -157.0,
                units_to_text: 0.001,
                to_unicode: None,
                encoding: None,
                single_byte_encoding: SingleByteEncoding::WinAnsi,
                single_byte_overrides: HashMap::new(),
                cid_widths: Vec::new(),
                cid_default_width: 600,
            })
        }
        Some(Base14Spec {
            family: Base14Family::Times,
            style,
        }) => {
            let mut widths = vec![0.0; 256];
            let base: &[(usize, i64)] = match style {
                Base14Style::Regular => &[
                    (32, 250),
                    (33, 333),
                    (34, 408),
                    (35, 500),
                    (36, 500),
                    (37, 833),
                    (38, 778),
                    (39, 180),
                    (40, 333),
                    (41, 333),
                    (42, 500),
                    (43, 564),
                    (44, 250),
                    (45, 333),
                    (46, 250),
                    (47, 278),
                    (48, 500),
                    (49, 500),
                    (50, 500),
                    (51, 500),
                    (52, 500),
                    (53, 500),
                    (54, 500),
                    (55, 500),
                    (56, 500),
                    (57, 500),
                    (58, 278),
                    (59, 278),
                    (60, 564),
                    (61, 564),
                    (62, 564),
                    (63, 444),
                    (64, 921),
                    (65, 722),
                    (66, 667),
                    (67, 667),
                    (68, 722),
                    (69, 611),
                    (70, 556),
                    (71, 722),
                    (72, 722),
                    (73, 333),
                    (74, 389),
                    (75, 722),
                    (76, 611),
                    (77, 889),
                    (78, 722),
                    (79, 722),
                    (80, 556),
                    (81, 722),
                    (82, 667),
                    (83, 556),
                    (84, 611),
                    (85, 722),
                    (86, 722),
                    (87, 944),
                    (88, 722),
                    (89, 722),
                    (90, 611),
                    (91, 333),
                    (92, 278),
                    (93, 333),
                    (94, 469),
                    (95, 500),
                    (96, 333),
                    (97, 444),
                    (98, 500),
                    (99, 444),
                    (100, 500),
                    (101, 444),
                    (102, 333),
                    (103, 500),
                    (104, 500),
                    (105, 278),
                    (106, 278),
                    (107, 500),
                    (108, 278),
                    (109, 778),
                    (110, 500),
                    (111, 500),
                    (112, 500),
                    (113, 500),
                    (114, 333),
                    (115, 389),
                    (116, 278),
                    (117, 500),
                    (118, 500),
                    (119, 722),
                    (120, 500),
                    (121, 500),
                    (122, 444),
                    (123, 480),
                    (124, 200),
                    (125, 480),
                    (126, 541),
                ],
                Base14Style::Bold => &[
                    (32, 250),
                    (33, 333),
                    (34, 555),
                    (35, 500),
                    (36, 500),
                    (37, 1000),
                    (38, 833),
                    (39, 278),
                    (40, 333),
                    (41, 333),
                    (42, 500),
                    (43, 570),
                    (44, 250),
                    (45, 333),
                    (46, 250),
                    (47, 278),
                    (48, 500),
                    (49, 500),
                    (50, 500),
                    (51, 500),
                    (52, 500),
                    (53, 500),
                    (54, 500),
                    (55, 500),
                    (56, 500),
                    (57, 500),
                    (58, 333),
                    (59, 333),
                    (60, 570),
                    (61, 570),
                    (62, 570),
                    (63, 500),
                    (64, 930),
                    (65, 722),
                    (66, 667),
                    (67, 722),
                    (68, 722),
                    (69, 667),
                    (70, 611),
                    (71, 778),
                    (72, 778),
                    (73, 389),
                    (74, 500),
                    (75, 778),
                    (76, 667),
                    (77, 944),
                    (78, 722),
                    (79, 778),
                    (80, 611),
                    (81, 778),
                    (82, 722),
                    (83, 556),
                    (84, 667),
                    (85, 722),
                    (86, 722),
                    (87, 1000),
                    (88, 722),
                    (89, 722),
                    (90, 667),
                    (91, 333),
                    (92, 278),
                    (93, 333),
                    (94, 581),
                    (95, 500),
                    (96, 333),
                    (97, 500),
                    (98, 556),
                    (99, 444),
                    (100, 556),
                    (101, 444),
                    (102, 333),
                    (103, 500),
                    (104, 556),
                    (105, 278),
                    (106, 333),
                    (107, 556),
                    (108, 278),
                    (109, 833),
                    (110, 556),
                    (111, 500),
                    (112, 556),
                    (113, 556),
                    (114, 444),
                    (115, 389),
                    (116, 333),
                    (117, 556),
                    (118, 500),
                    (119, 722),
                    (120, 500),
                    (121, 500),
                    (122, 444),
                    (123, 394),
                    (124, 220),
                    (125, 394),
                    (126, 520),
                ],
                Base14Style::Italic => &[
                    (32, 250),
                    (33, 333),
                    (34, 420),
                    (35, 500),
                    (36, 500),
                    (37, 833),
                    (38, 778),
                    (39, 214),
                    (40, 333),
                    (41, 333),
                    (42, 500),
                    (43, 675),
                    (44, 250),
                    (45, 333),
                    (46, 250),
                    (47, 278),
                    (48, 500),
                    (49, 500),
                    (50, 500),
                    (51, 500),
                    (52, 500),
                    (53, 500),
                    (54, 500),
                    (55, 500),
                    (56, 500),
                    (57, 500),
                    (58, 333),
                    (59, 333),
                    (60, 675),
                    (61, 675),
                    (62, 675),
                    (63, 500),
                    (64, 920),
                    (65, 611),
                    (66, 611),
                    (67, 667),
                    (68, 722),
                    (69, 611),
                    (70, 611),
                    (71, 722),
                    (72, 722),
                    (73, 333),
                    (74, 444),
                    (75, 667),
                    (76, 556),
                    (77, 833),
                    (78, 667),
                    (79, 722),
                    (80, 611),
                    (81, 722),
                    (82, 611),
                    (83, 500),
                    (84, 556),
                    (85, 722),
                    (86, 611),
                    (87, 833),
                    (88, 611),
                    (89, 556),
                    (90, 556),
                    (91, 389),
                    (92, 278),
                    (93, 389),
                    (94, 422),
                    (95, 500),
                    (96, 333),
                    (97, 500),
                    (98, 500),
                    (99, 444),
                    (100, 500),
                    (101, 444),
                    (102, 278),
                    (103, 500),
                    (104, 500),
                    (105, 278),
                    (106, 278),
                    (107, 444),
                    (108, 278),
                    (109, 722),
                    (110, 500),
                    (111, 500),
                    (112, 500),
                    (113, 500),
                    (114, 389),
                    (115, 389),
                    (116, 278),
                    (117, 500),
                    (118, 444),
                    (119, 667),
                    (120, 444),
                    (121, 444),
                    (122, 389),
                    (123, 400),
                    (124, 275),
                    (125, 400),
                    (126, 541),
                ],
                Base14Style::BoldItalic => &[
                    (32, 250),
                    (33, 389),
                    (34, 555),
                    (35, 500),
                    (36, 500),
                    (37, 833),
                    (38, 778),
                    (39, 278),
                    (40, 333),
                    (41, 333),
                    (42, 500),
                    (43, 570),
                    (44, 250),
                    (45, 333),
                    (46, 250),
                    (47, 278),
                    (48, 500),
                    (49, 500),
                    (50, 500),
                    (51, 500),
                    (52, 500),
                    (53, 500),
                    (54, 500),
                    (55, 500),
                    (56, 500),
                    (57, 500),
                    (58, 333),
                    (59, 333),
                    (60, 570),
                    (61, 570),
                    (62, 570),
                    (63, 500),
                    (64, 832),
                    (65, 667),
                    (66, 667),
                    (67, 667),
                    (68, 722),
                    (69, 667),
                    (70, 667),
                    (71, 722),
                    (72, 778),
                    (73, 389),
                    (74, 500),
                    (75, 667),
                    (76, 611),
                    (77, 889),
                    (78, 722),
                    (79, 722),
                    (80, 611),
                    (81, 722),
                    (82, 667),
                    (83, 556),
                    (84, 611),
                    (85, 722),
                    (86, 667),
                    (87, 889),
                    (88, 667),
                    (89, 611),
                    (90, 611),
                    (91, 333),
                    (92, 278),
                    (93, 333),
                    (94, 570),
                    (95, 500),
                    (96, 333),
                    (97, 500),
                    (98, 500),
                    (99, 444),
                    (100, 500),
                    (101, 444),
                    (102, 333),
                    (103, 500),
                    (104, 556),
                    (105, 278),
                    (106, 278),
                    (107, 500),
                    (108, 278),
                    (109, 778),
                    (110, 556),
                    (111, 500),
                    (112, 500),
                    (113, 500),
                    (114, 389),
                    (115, 389),
                    (116, 278),
                    (117, 556),
                    (118, 444),
                    (119, 667),
                    (120, 500),
                    (121, 444),
                    (122, 389),
                    (123, 348),
                    (124, 220),
                    (125, 348),
                    (126, 570),
                ],
            };
            for (code, width) in base {
                widths[*code] = *width as f64;
            }
            Some(FontMetrics {
                first_char: 0,
                widths,
                ascent: 683.0,
                descent: -217.0,
                units_to_text: 0.001,
                to_unicode: None,
                encoding: None,
                single_byte_encoding: SingleByteEncoding::WinAnsi,
                single_byte_overrides: HashMap::new(),
                cid_widths: Vec::new(),
                cid_default_width: 500,
            })
        }
        Some(Base14Spec {
            family: Base14Family::Symbol,
            ..
        }) => {
            let mut widths = vec![500.0; 256];
            widths.iter_mut().take(32).for_each(|w| *w = 0.0);
            Some(FontMetrics {
                first_char: 0,
                widths,
                ascent: 700.0,
                descent: -200.0,
                units_to_text: 0.001,
                to_unicode: None,
                encoding: None,
                single_byte_encoding: SingleByteEncoding::WinAnsi,
                single_byte_overrides: HashMap::new(),
                cid_widths: Vec::new(),
                cid_default_width: 500,
            })
        }
        Some(Base14Spec {
            family: Base14Family::ZapfDingbats,
            ..
        }) => {
            let mut widths = vec![500.0; 256];
            widths.iter_mut().take(32).for_each(|w| *w = 0.0);
            Some(FontMetrics {
                first_char: 0,
                widths,
                ascent: 700.0,
                descent: -200.0,
                units_to_text: 0.001,
                to_unicode: None,
                encoding: None,
                single_byte_encoding: SingleByteEncoding::WinAnsi,
                single_byte_overrides: HashMap::new(),
                cid_widths: Vec::new(),
                cid_default_width: 500,
            })
        }
        _ => None,
    }
}

#[derive(Clone, Debug)]
struct CidWidthRange {
    start: u32,
    end: u32,
    width: i64,
}

fn parse_cid_widths(arr: &[Object]) -> Vec<CidWidthRange> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < arr.len() {
        let start = match arr[i].as_i64() {
            Some(v) => v as u32,
            None => {
                i += 1;
                continue;
            }
        };
        if i + 1 >= arr.len() {
            break;
        }
        match &arr[i + 1] {
            Object::Array(widths) => {
                let mut cid = start;
                for w in widths {
                    if let Some(wv) = w.as_f64() {
                        out.push(CidWidthRange {
                            start: cid,
                            end: cid,
                            width: wv.round() as i64,
                        });
                    }
                    cid += 1;
                }
                i += 2;
            }
            _ if arr[i + 1].as_i64().is_some() => {
                if i + 2 >= arr.len() {
                    break;
                }
                let end = arr[i + 1].as_i64().unwrap_or(start as i64) as u32;
                let w = arr[i + 2].as_f64().unwrap_or(0.0).round() as i64;
                out.push(CidWidthRange {
                    start,
                    end,
                    width: w,
                });
                i += 3;
            }
            _ => {
                i += 1;
            }
        }
    }
    out
}

fn resolve_cid_width_array(doc: &PdfDoc, arr: &[Object]) -> Vec<Object> {
    arr.iter()
        .map(|entry| {
            let resolved = doc.resolve(entry).clone();
            if let Object::Array(values) = &resolved {
                Object::Array(
                    values
                        .iter()
                        .map(|value| doc.resolve(value).clone())
                        .collect(),
                )
            } else {
                resolved
            }
        })
        .collect()
}

#[derive(Clone, Debug)]
struct CMap {
    codespaces: Vec<CodeSpaceRange>,
    map: HashMap<u32, String>,
}

#[derive(Clone, Debug)]
struct EncodingMap {
    codespaces: Vec<CodeSpaceRange>,
    map: HashMap<u32, u32>,
}

#[derive(Clone, Debug)]
struct CodeSpaceRange {
    start: u32,
    end: u32,
    len: usize,
}

#[cfg(test)]
#[derive(Debug)]
struct DecodedGlyph {
    code: u32,
    text: String,
}

fn strip_replacement_chars(text: &str) -> String {
    text.chars().filter(|&ch| ch != '\u{FFFD}').collect()
}

#[cfg(test)]
fn decode_text(font: &FontMetrics, bytes: &[u8]) -> Vec<DecodedGlyph> {
    let mut out = Vec::new();
    for_each_decoded_glyph(font, bytes, |code, text| {
        out.push(DecodedGlyph {
            code,
            text: text.to_string(),
        });
    });
    out
}

fn for_each_decoded_glyph<F>(font: &FontMetrics, bytes: &[u8], mut f: F)
where
    F: FnMut(u32, &str),
{
    let mut i = 0;
    while i < bytes.len() {
        let (code, len) = if let Some(enc) = &font.encoding {
            cmap_next_code_with(&enc.codespaces, bytes, i).unwrap_or((bytes[i] as u32, 1))
        } else if let Some(cmap) = &font.to_unicode {
            cmap_next_code_with(&cmap.codespaces, bytes, i).unwrap_or((bytes[i] as u32, 1))
        } else {
            (bytes[i] as u32, 1)
        };
        let cid = if let Some(enc) = &font.encoding {
            enc.map.get(&code).cloned().unwrap_or(code)
        } else {
            code
        };
        if let Some(cmap) = &font.to_unicode {
            if let Some(mapped) = cmap.map.get(&code) {
                if mapped.contains('\u{FFFD}') {
                    let stripped = strip_replacement_chars(mapped);
                    f(cid, stripped.as_str());
                } else {
                    f(cid, mapped.as_str());
                }
            } else {
                let mut utf8 = [0u8; 4];
                if let Some(ch) = decode_single_byte_char(font, code) {
                    f(cid, ch.encode_utf8(&mut utf8));
                } else {
                    f(cid, "");
                }
            }
        } else {
            let mut utf8 = [0u8; 4];
            if let Some(ch) = decode_single_byte_char(font, code) {
                f(cid, ch.encode_utf8(&mut utf8));
            } else {
                f(cid, "");
            }
        }
        i += len;
    }
}

fn decode_single_byte_char(font: &FontMetrics, code: u32) -> Option<char> {
    if let Some(mapped) = font.single_byte_overrides.get(&code) {
        return Some(*mapped);
    }
    single_byte_char(font.single_byte_encoding, code)
}

fn single_byte_char(encoding: SingleByteEncoding, code: u32) -> Option<char> {
    match encoding {
        SingleByteEncoding::WinAnsi => win_ansi_char(code),
        SingleByteEncoding::MacRoman => table_char(&MAC_ROMAN, code),
    }
}

fn win_ansi_char(code: u32) -> Option<char> {
    table_char(&WIN_ANSI, code)
}

fn table_char(table: &[u16; 256], code: u32) -> Option<char> {
    if code > 0xFF {
        return None;
    }
    let mapped = table[code as usize];
    if mapped == 0 {
        return None;
    }
    char::from_u32(mapped as u32)
}

fn cmap_next_code_with(
    codespaces: &[CodeSpaceRange],
    bytes: &[u8],
    offset: usize,
) -> Option<(u32, usize)> {
    for len in 1..=4 {
        if offset + len > bytes.len() {
            break;
        }
        let code = bytes_to_u32(&bytes[offset..offset + len]);
        if codespaces
            .iter()
            .any(|r| r.len == len && code >= r.start && code <= r.end)
        {
            return Some((code, len));
        }
    }
    None
}

fn bytes_to_u32(bytes: &[u8]) -> u32 {
    let mut v = 0u32;
    for &b in bytes {
        v = (v << 8) | b as u32;
    }
    v
}

fn parse_cmap(data: &[u8]) -> CMap {
    let mut tokens = CMapTokenizer::new(data);
    let mut codespaces = Vec::new();
    let mut map = HashMap::new();
    while let Some(tok) = tokens.next() {
        match tok.as_str() {
            "begincodespacerange" => {
                while let Some(t) = tokens.next() {
                    if t == "endcodespacerange" {
                        break;
                    }
                    let start = parse_hex_token(&t);
                    let end = parse_hex_token(tokens.next().as_deref().unwrap_or(""));
                    if !start.is_empty() && !end.is_empty() && start.len() == end.len() {
                        let len = start.len();
                        codespaces.push(CodeSpaceRange {
                            start: bytes_to_u32(&start),
                            end: bytes_to_u32(&end),
                            len,
                        });
                    }
                }
            }
            "beginbfchar" => {
                while let Some(t) = tokens.next() {
                    if t == "endbfchar" {
                        break;
                    }
                    let src = parse_hex_token(&t);
                    let dst = parse_hex_token(tokens.next().as_deref().unwrap_or(""));
                    if let Some(s) = utf16be_to_string(&dst) {
                        map.insert(bytes_to_u32(&src), s);
                    }
                }
            }
            "beginbfrange" => {
                while let Some(t) = tokens.next() {
                    if t == "endbfrange" {
                        break;
                    }
                    let start = parse_hex_token(&t);
                    let end = parse_hex_token(tokens.next().as_deref().unwrap_or(""));
                    let next = tokens.next().unwrap_or_default();
                    if next.starts_with('[') {
                        let mut code = bytes_to_u32(&start);
                        let end_code = bytes_to_u32(&end);
                        let mut cur = next;
                        loop {
                            if cur.ends_with(']') {
                                let inner = cur.trim_end_matches(']').to_string();
                                if !inner.is_empty() {
                                    let dst = parse_hex_token(&inner);
                                    if let Some(s) = utf16be_to_string(&dst) {
                                        map.insert(code, s);
                                    }
                                }
                                break;
                            }
                            let dst = parse_hex_token(&cur);
                            if let Some(s) = utf16be_to_string(&dst) {
                                map.insert(code, s);
                            }
                            code += 1;
                            if code > end_code {
                                break;
                            }
                            cur = tokens.next().unwrap_or_default();
                        }
                    } else {
                        let dst = parse_hex_token(&next);
                        let mut code = bytes_to_u32(&start);
                        let end_code = bytes_to_u32(&end);
                        if let Some(s) = utf16be_to_string(&dst) {
                            let mut chars = s.chars();
                            let mut base = chars.next().unwrap_or('\u{FFFD}') as u32;
                            while code <= end_code {
                                if let Some(ch) = char::from_u32(base) {
                                    map.insert(code, ch.to_string());
                                }
                                code += 1;
                                base += 1;
                            }
                        }
                    }
                }
            }
            _ => {}
        }
    }

    if codespaces.is_empty() {
        codespaces.push(CodeSpaceRange {
            start: 0x00,
            end: 0xFF,
            len: 1,
        });
    }

    CMap { codespaces, map }
}

fn parse_encoding(doc: &PdfDoc, obj: &Object) -> Option<EncodingMap> {
    match doc.resolve(obj) {
        Object::Name(name) => {
            if name == "Identity-H" || name == "Identity-V" {
                return Some(EncodingMap {
                    codespaces: vec![CodeSpaceRange {
                        start: 0x0000,
                        end: 0xFFFF,
                        len: 2,
                    }],
                    map: HashMap::new(),
                });
            }
            None
        }
        Object::Stream { .. } => {
            let data = decode_stream(doc, obj)?;
            Some(parse_encoding_cmap(&data))
        }
        _ => None,
    }
}

fn parse_single_byte_encoding(
    doc: &PdfDoc,
    obj: &Object,
) -> Option<(SingleByteEncoding, HashMap<u32, char>)> {
    match doc.resolve(obj) {
        Object::Name(name) => single_byte_encoding_from_name(name).map(|enc| (enc, HashMap::new())),
        Object::Dictionary(dict) => {
            let (base_encoding, mut overrides) = dict
                .get("BaseEncoding")
                .and_then(|base| parse_single_byte_encoding(doc, base))
                .unwrap_or((SingleByteEncoding::WinAnsi, HashMap::new()));
            if let Some(differences) = dict.get("Differences") {
                parse_single_byte_differences(doc, differences, &mut overrides);
            }
            Some((base_encoding, overrides))
        }
        _ => None,
    }
}

fn parse_single_byte_differences(doc: &PdfDoc, obj: &Object, overrides: &mut HashMap<u32, char>) {
    let Some(differences) = doc.resolve(obj).as_array() else {
        return;
    };
    let mut next_code: Option<u32> = None;
    for entry in differences {
        match doc.resolve(entry) {
            Object::Integer(code) if (0..=255).contains(code) => {
                next_code = Some(*code as u32);
            }
            Object::Name(name) => {
                if let Some(code) = next_code {
                    if let Some(mapped) = single_byte_glyph_name_to_char(name) {
                        overrides.insert(code, mapped);
                    }
                    next_code = if code < 255 { Some(code + 1) } else { None };
                }
            }
            _ => {}
        }
    }
}

fn single_byte_glyph_name_to_char(name: &str) -> Option<char> {
    if name.eq_ignore_ascii_case("space")
        || name.eq_ignore_ascii_case("nonbreakingspace")
        || name.eq_ignore_ascii_case("nbspace")
    {
        return Some(' ');
    }
    None
}

fn single_byte_encoding_from_name(name: &str) -> Option<SingleByteEncoding> {
    match name {
        "WinAnsiEncoding" => Some(SingleByteEncoding::WinAnsi),
        "MacRomanEncoding" => Some(SingleByteEncoding::MacRoman),
        _ => None,
    }
}

fn parse_encoding_cmap(data: &[u8]) -> EncodingMap {
    let mut tokens = CMapTokenizer::new(data);
    let mut codespaces = Vec::new();
    let mut map = HashMap::new();
    while let Some(tok) = tokens.next() {
        match tok.as_str() {
            "begincodespacerange" => {
                while let Some(t) = tokens.next() {
                    if t == "endcodespacerange" {
                        break;
                    }
                    let start = parse_hex_token(&t);
                    let end = parse_hex_token(tokens.next().as_deref().unwrap_or(""));
                    if !start.is_empty() && !end.is_empty() && start.len() == end.len() {
                        let len = start.len();
                        codespaces.push(CodeSpaceRange {
                            start: bytes_to_u32(&start),
                            end: bytes_to_u32(&end),
                            len,
                        });
                    }
                }
            }
            "beginbfchar" => {
                while let Some(t) = tokens.next() {
                    if t == "endbfchar" {
                        break;
                    }
                    let src = parse_hex_token(&t);
                    let dst = parse_hex_token(tokens.next().as_deref().unwrap_or(""));
                    if !src.is_empty() && !dst.is_empty() {
                        map.insert(bytes_to_u32(&src), bytes_to_u32(&dst));
                    }
                }
            }
            "beginbfrange" => {
                while let Some(t) = tokens.next() {
                    if t == "endbfrange" {
                        break;
                    }
                    let start = parse_hex_token(&t);
                    let end = parse_hex_token(tokens.next().as_deref().unwrap_or(""));
                    let next = tokens.next().unwrap_or_default();
                    let start_code = bytes_to_u32(&start);
                    let end_code = bytes_to_u32(&end);
                    if next.starts_with('[') {
                        let mut code = start_code;
                        let mut cur = next;
                        loop {
                            if cur.ends_with(']') {
                                let inner = cur.trim_end_matches(']').to_string();
                                if !inner.is_empty() {
                                    let dst = parse_hex_token(&inner);
                                    map.insert(code, bytes_to_u32(&dst));
                                }
                                break;
                            }
                            let dst = parse_hex_token(&cur);
                            map.insert(code, bytes_to_u32(&dst));
                            code += 1;
                            if code > end_code {
                                break;
                            }
                            cur = tokens.next().unwrap_or_default();
                        }
                    } else {
                        let dst = parse_hex_token(&next);
                        let mut code = start_code;
                        let mut cid = bytes_to_u32(&dst);
                        while code <= end_code {
                            map.insert(code, cid);
                            code += 1;
                            cid += 1;
                        }
                    }
                }
            }
            _ => {}
        }
    }

    if codespaces.is_empty() {
        codespaces.push(CodeSpaceRange {
            start: 0x00,
            end: 0xFF,
            len: 1,
        });
    }

    EncodingMap { codespaces, map }
}

fn parse_hex_token(token: &str) -> Vec<u8> {
    let t = token.trim();
    if !t.starts_with('<') || !t.ends_with('>') {
        return Vec::new();
    }
    let inner = &t[1..t.len() - 1];
    let mut out = Vec::new();
    let mut chars = inner.chars();
    while let (Some(a), Some(b)) = (chars.next(), chars.next()) {
        let hex = format!("{a}{b}");
        if let Ok(v) = u8::from_str_radix(&hex, 16) {
            out.push(v);
        }
    }
    out
}

fn utf16be_to_string(bytes: &[u8]) -> Option<String> {
    if bytes.is_empty() || bytes.len() % 2 != 0 {
        return None;
    }
    let mut u16s = Vec::with_capacity(bytes.len() / 2);
    let mut it = bytes.iter();
    while let (Some(a), Some(b)) = (it.next(), it.next()) {
        u16s.push(u16::from_be_bytes([*a, *b]));
    }
    String::from_utf16(&u16s).ok()
}

struct CMapTokenizer<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> CMapTokenizer<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    fn next(&mut self) -> Option<String> {
        while self.pos < self.data.len() && self.data[self.pos].is_ascii_whitespace() {
            self.pos += 1;
        }
        if self.pos >= self.data.len() {
            return None;
        }
        let b = self.data[self.pos];
        if b == b'<' {
            let start = self.pos;
            self.pos += 1;
            while self.pos < self.data.len() && self.data[self.pos] != b'>' {
                self.pos += 1;
            }
            if self.pos < self.data.len() {
                self.pos += 1;
            }
            return Some(String::from_utf8_lossy(&self.data[start..self.pos]).to_string());
        }
        if b == b'[' {
            let start = self.pos;
            self.pos += 1;
            while self.pos < self.data.len() && self.data[self.pos].is_ascii_whitespace() {
                self.pos += 1;
            }
            return Some(String::from_utf8_lossy(&self.data[start..self.pos]).to_string());
        }
        if b == b']' {
            self.pos += 1;
            return Some("]".to_string());
        }
        let start = self.pos;
        while self.pos < self.data.len()
            && !self.data[self.pos].is_ascii_whitespace()
            && self.data[self.pos] != b'['
            && self.data[self.pos] != b']'
        {
            self.pos += 1;
        }
        Some(String::from_utf8_lossy(&self.data[start..self.pos]).to_string())
    }
}

const WIN_ANSI: [u16; 256] = [
    0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
    0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
    0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0020, 0x0021, 0x0022, 0x0023,
    0x0024, 0x0025, 0x0026, 0x0027, 0x0028, 0x0029, 0x002A, 0x002B, 0x002C, 0x002D, 0x002E, 0x002F,
    0x0030, 0x0031, 0x0032, 0x0033, 0x0034, 0x0035, 0x0036, 0x0037, 0x0038, 0x0039, 0x003A, 0x003B,
    0x003C, 0x003D, 0x003E, 0x003F, 0x0040, 0x0041, 0x0042, 0x0043, 0x0044, 0x0045, 0x0046, 0x0047,
    0x0048, 0x0049, 0x004A, 0x004B, 0x004C, 0x004D, 0x004E, 0x004F, 0x0050, 0x0051, 0x0052, 0x0053,
    0x0054, 0x0055, 0x0056, 0x0057, 0x0058, 0x0059, 0x005A, 0x005B, 0x005C, 0x005D, 0x005E, 0x005F,
    0x0060, 0x0061, 0x0062, 0x0063, 0x0064, 0x0065, 0x0066, 0x0067, 0x0068, 0x0069, 0x006A, 0x006B,
    0x006C, 0x006D, 0x006E, 0x006F, 0x0070, 0x0071, 0x0072, 0x0073, 0x0074, 0x0075, 0x0076, 0x0077,
    0x0078, 0x0079, 0x007A, 0x007B, 0x007C, 0x007D, 0x007E, 0x0000, 0x20AC, 0x0000, 0x201A, 0x0192,
    0x201E, 0x2026, 0x2020, 0x2021, 0x02C6, 0x2030, 0x0160, 0x2039, 0x0152, 0x0000, 0x017D, 0x0000,
    0x0000, 0x2018, 0x2019, 0x201C, 0x201D, 0x2022, 0x2013, 0x2014, 0x02DC, 0x2122, 0x0161, 0x203A,
    0x0153, 0x0000, 0x017E, 0x0178, 0x00A0, 0x00A1, 0x00A2, 0x00A3, 0x00A4, 0x00A5, 0x00A6, 0x00A7,
    0x00A8, 0x00A9, 0x00AA, 0x00AB, 0x00AC, 0x00AD, 0x00AE, 0x00AF, 0x00B0, 0x00B1, 0x00B2, 0x00B3,
    0x00B4, 0x00B5, 0x00B6, 0x00B7, 0x00B8, 0x00B9, 0x00BA, 0x00BB, 0x00BC, 0x00BD, 0x00BE, 0x00BF,
    0x00C0, 0x00C1, 0x00C2, 0x00C3, 0x00C4, 0x00C5, 0x00C6, 0x00C7, 0x00C8, 0x00C9, 0x00CA, 0x00CB,
    0x00CC, 0x00CD, 0x00CE, 0x00CF, 0x00D0, 0x00D1, 0x00D2, 0x00D3, 0x00D4, 0x00D5, 0x00D6, 0x00D7,
    0x00D8, 0x00D9, 0x00DA, 0x00DB, 0x00DC, 0x00DD, 0x00DE, 0x00DF, 0x00E0, 0x00E1, 0x00E2, 0x00E3,
    0x00E4, 0x00E5, 0x00E6, 0x00E7, 0x00E8, 0x00E9, 0x00EA, 0x00EB, 0x00EC, 0x00ED, 0x00EE, 0x00EF,
    0x00F0, 0x00F1, 0x00F2, 0x00F3, 0x00F4, 0x00F5, 0x00F6, 0x00F7, 0x00F8, 0x00F9, 0x00FA, 0x00FB,
    0x00FC, 0x00FD, 0x00FE, 0x00FF,
];

const MAC_ROMAN: [u16; 256] = [
    0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
    0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
    0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0020, 0x0021, 0x0022, 0x0023,
    0x0024, 0x0025, 0x0026, 0x0027, 0x0028, 0x0029, 0x002A, 0x002B, 0x002C, 0x002D, 0x002E, 0x002F,
    0x0030, 0x0031, 0x0032, 0x0033, 0x0034, 0x0035, 0x0036, 0x0037, 0x0038, 0x0039, 0x003A, 0x003B,
    0x003C, 0x003D, 0x003E, 0x003F, 0x0040, 0x0041, 0x0042, 0x0043, 0x0044, 0x0045, 0x0046, 0x0047,
    0x0048, 0x0049, 0x004A, 0x004B, 0x004C, 0x004D, 0x004E, 0x004F, 0x0050, 0x0051, 0x0052, 0x0053,
    0x0054, 0x0055, 0x0056, 0x0057, 0x0058, 0x0059, 0x005A, 0x005B, 0x005C, 0x005D, 0x005E, 0x005F,
    0x0060, 0x0061, 0x0062, 0x0063, 0x0064, 0x0065, 0x0066, 0x0067, 0x0068, 0x0069, 0x006A, 0x006B,
    0x006C, 0x006D, 0x006E, 0x006F, 0x0070, 0x0071, 0x0072, 0x0073, 0x0074, 0x0075, 0x0076, 0x0077,
    0x0078, 0x0079, 0x007A, 0x007B, 0x007C, 0x007D, 0x007E, 0x0000, 0x00C4, 0x00C5, 0x00C7, 0x00C9,
    0x00D1, 0x00D6, 0x00DC, 0x00E1, 0x00E0, 0x00E2, 0x00E4, 0x00E3, 0x00E5, 0x00E7, 0x00E9, 0x00E8,
    0x00EA, 0x00EB, 0x00ED, 0x00EC, 0x00EE, 0x00EF, 0x00F1, 0x00F3, 0x00F2, 0x00F4, 0x00F6, 0x00F5,
    0x00FA, 0x00F9, 0x00FB, 0x00FC, 0x2020, 0x00B0, 0x00A2, 0x00A3, 0x00A7, 0x2022, 0x00B6, 0x00DF,
    0x00AE, 0x00A9, 0x2122, 0x00B4, 0x00A8, 0x2260, 0x00C6, 0x00D8, 0x221E, 0x00B1, 0x2264, 0x2265,
    0x00A5, 0x00B5, 0x2202, 0x2211, 0x220F, 0x03C0, 0x222B, 0x00AA, 0x00BA, 0x03A9, 0x00E6, 0x00F8,
    0x00BF, 0x00A1, 0x00AC, 0x221A, 0x0192, 0x2248, 0x2206, 0x00AB, 0x00BB, 0x2026, 0x00A0, 0x00C0,
    0x00C3, 0x00D5, 0x0152, 0x0153, 0x2013, 0x2014, 0x201C, 0x201D, 0x2018, 0x2019, 0x00F7, 0x25CA,
    0x00FF, 0x0178, 0x2044, 0x20AC, 0x2039, 0x203A, 0xFB01, 0xFB02, 0x2021, 0x00B7, 0x201A, 0x201E,
    0x2030, 0x00C2, 0x00CA, 0x00C1, 0x00CB, 0x00C8, 0x00CD, 0x00CE, 0x00CF, 0x00CC, 0x00D3, 0x00D4,
    0xF8FF, 0x00D2, 0x00DA, 0x00DB, 0x00D9, 0x0131, 0x02C6, 0x02DC, 0x00AF, 0x02D8, 0x02D9, 0x02DA,
    0x00B8, 0x02DD, 0x02DB, 0x02C7,
];

struct ContentTokenizer<'a> {
    lexer: Lexer<'a>,
}

impl<'a> ContentTokenizer<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self {
            lexer: Lexer::new(data),
        }
    }

    fn next_op_into(&mut self, operands: &mut Vec<Object>) -> Option<String> {
        operands.clear();
        while let Some(tok) = self.lexer.next_token() {
            match tok {
                Token::Keyword(op) => {
                    if op == "BI" {
                        self.skip_inline_image();
                        operands.clear();
                        continue;
                    }
                    return Some(op);
                }
                _ => {
                    if let Some(obj) = self.parse_object_from_token(tok) {
                        operands.push(obj);
                    }
                }
            }
        }
        None
    }

    fn parse_object_from_token(&mut self, tok: Token) -> Option<Object> {
        match tok {
            Token::Null => Some(Object::Null),
            Token::Boolean(v) => Some(Object::Boolean(v)),
            Token::Integer(v) => Some(Object::Integer(v)),
            Token::Real(v) => Some(Object::Real(v)),
            Token::String(v) => Some(Object::String(v)),
            Token::HexString(v) => Some(Object::String(v)),
            Token::Name(v) => Some(Object::Name(v)),
            Token::ArrayStart => Some(Object::Array(self.parse_array())),
            _ => None,
        }
    }

    fn parse_array(&mut self) -> Vec<Object> {
        let mut items = Vec::new();
        while let Some(tok) = self.lexer.next_token() {
            match tok {
                Token::ArrayEnd => break,
                _ => {
                    if let Some(obj) = self.parse_object_from_token(tok) {
                        items.push(obj);
                    }
                }
            }
        }
        items
    }

    fn skip_inline_image(&mut self) {
        while let Some(tok) = self.lexer.next_token() {
            if let Token::Keyword(op) = tok
                && op == "ID"
            {
                self.lexer.skip_inline_image_data();
                break;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn canonical_base14_aliases_resolve() {
        let spec =
            canonical_base14_spec("ABCDEE+TimesNewRomanPS-BoldMT").expect("expected Times alias");
        assert_eq!(spec.family, Base14Family::Times);
        assert_eq!(spec.style, Base14Style::Bold);

        let spec =
            canonical_base14_spec("CourierNewPS-BoldItalicMT").expect("expected Courier alias");
        assert_eq!(spec.family, Base14Family::Courier);
        assert_eq!(spec.style, Base14Style::BoldItalic);

        let spec = canonical_base14_spec("CourierNew").expect("expected Courier alias");
        assert_eq!(spec.family, Base14Family::Courier);
        assert_eq!(spec.style, Base14Style::Regular);

        let spec = canonical_base14_spec("CourierNew,Bold").expect("expected Courier alias");
        assert_eq!(spec.family, Base14Family::Courier);
        assert_eq!(spec.style, Base14Style::Bold);

        let spec = canonical_base14_spec("SymbolMT").expect("expected Symbol alias");
        assert_eq!(spec.family, Base14Family::Symbol);

        let spec = canonical_base14_spec("ITCZapfDingbatsStd").expect("expected Zapf alias");
        assert_eq!(spec.family, Base14Family::ZapfDingbats);
    }

    #[test]
    fn base14_metrics_exist_for_all_supported_families() {
        assert!(base14_metrics("Helvetica").is_some());
        assert!(base14_metrics("Courier").is_some());
        assert!(base14_metrics("Times-Roman").is_some());
        assert!(base14_metrics("Times-BoldItalic").is_some());
        assert!(base14_metrics("Symbol").is_some());
        assert!(base14_metrics("ZapfDingbats").is_some());
    }

    #[test]
    fn translated_ctm_is_not_scaled_by_font_size() {
        let mut text_state = TextState::new();
        text_state.font_size = 9.0;
        text_state.text_matrix = Matrix::translate(8.200005, 1.197998);

        let ctm = Matrix::translate(177.571, 620.667);
        let font_matrix = Matrix {
            a: text_state.font_size * (text_state.horiz_scaling / 100.0),
            b: 0.0,
            c: 0.0,
            d: text_state.font_size,
            e: 0.0,
            f: text_state.rise,
        };

        let text_space = text_state.text_matrix.multiply(font_matrix);

        let corrected = ctm.multiply(text_space);
        let (cx, cy) = corrected.apply_to_point(0.0, 0.0);
        assert!(
            (cx - 185.771005).abs() < 1e-6,
            "unexpected corrected x {}",
            cx
        );
        assert!(
            (cy - 621.864998).abs() < 1e-6,
            "unexpected corrected y {}",
            cy
        );
        assert!(cx < 1000.0 && cy < 1000.0, "corrected origin is off-page");

        let legacy = text_space.multiply(ctm);
        let (lx, ly) = legacy.apply_to_point(0.0, 0.0);
        assert!(
            lx > 1000.0 && ly > 1000.0,
            "legacy order no longer demonstrates scaled translation: ({}, {})",
            lx,
            ly
        );
    }

    #[test]
    fn parse_cid_widths_handles_real_width_values() {
        let widths = vec![
            Object::Integer(10),
            Object::Array(vec![Object::Real(556.15234), Object::Real(277.83203)]),
            Object::Integer(20),
            Object::Integer(22),
            Object::Real(500.51),
        ];

        let parsed = parse_cid_widths(&widths);
        assert_eq!(parsed.len(), 3);

        assert_eq!(parsed[0].start, 10);
        assert_eq!(parsed[0].end, 10);
        assert_eq!(parsed[0].width, 556);

        assert_eq!(parsed[1].start, 11);
        assert_eq!(parsed[1].end, 11);
        assert_eq!(parsed[1].width, 278);

        assert_eq!(parsed[2].start, 20);
        assert_eq!(parsed[2].end, 22);
        assert_eq!(parsed[2].width, 501);
    }

    #[test]
    fn decode_text_keeps_multi_char_mappings_and_drops_replacement_chars() {
        let mut map = HashMap::new();
        map.insert(0x0001, "Sta".to_string());
        map.insert(0x0002, "\u{FFFD}".to_string());
        map.insert(0x0003, "a\u{FFFD}b".to_string());

        let font = FontMetrics {
            first_char: 0,
            widths: Vec::new(),
            ascent: 800.0,
            descent: -200.0,
            units_to_text: 0.001,
            to_unicode: Some(CMap {
                codespaces: vec![CodeSpaceRange {
                    start: 0x0000,
                    end: 0xFFFF,
                    len: 2,
                }],
                map,
            }),
            encoding: None,
            single_byte_encoding: SingleByteEncoding::WinAnsi,
            single_byte_overrides: HashMap::new(),
            cid_widths: Vec::new(),
            cid_default_width: 1000,
        };

        let decoded = decode_text(&font, &[0x00, 0x01, 0x00, 0x02, 0x00, 0x03]);
        assert_eq!(decoded.len(), 3);
        assert_eq!(decoded[0].code, 0x0001);
        assert_eq!(decoded[0].text, "Sta");
        assert_eq!(decoded[1].code, 0x0002);
        assert!(decoded[1].text.is_empty());
        assert_eq!(decoded[2].code, 0x0003);
        assert_eq!(decoded[2].text, "ab");
    }

    fn make_rect(min_x: f64, min_y: f64, max_x: f64, max_y: f64) -> Rect {
        Rect {
            min_x,
            min_y,
            max_x,
            max_y,
        }
    }

    fn make_visible_text_state(fill_rgb: (f64, f64, f64)) -> TextState {
        let mut state = TextState::new();
        state.fill_rgb = Some(fill_rgb);
        state.render_mode = 0;
        state
    }

    #[test]
    fn text_visibility_returns_true_when_white_contrast_already_sufficient() {
        let text_state = make_visible_text_state((0.0, 0.0, 0.0));
        let bbox = make_rect(0.0, 0.0, 1.0, 1.0);
        let fills = vec![PaintedFill {
            bbox: make_rect(0.0, 0.0, 1.0, 1.0),
            bg_luminance: Some(0.0),
        }];
        assert!(text_is_visible_at_bbox(&text_state, bbox, &fills));
    }

    #[test]
    fn text_visibility_returns_true_when_fill_raises_contrast_above_threshold() {
        let text_state = make_visible_text_state((1.0, 1.0, 1.0));
        let bbox = make_rect(0.0, 0.0, 1.0, 1.0);
        let fills = vec![PaintedFill {
            bbox: make_rect(0.0, 0.0, 1.0, 1.0),
            bg_luminance: Some(0.0),
        }];
        assert!(text_is_visible_at_bbox(&text_state, bbox, &fills));
    }

    #[test]
    fn text_visibility_returns_false_when_all_contrasts_below_threshold() {
        let text_state = make_visible_text_state((1.0, 1.0, 1.0));
        let bbox = make_rect(0.0, 0.0, 1.0, 1.0);
        let fills = vec![PaintedFill {
            bbox: make_rect(0.0, 0.0, 1.0, 1.0),
            bg_luminance: Some(0.99),
        }];
        assert!(!text_is_visible_at_bbox(&text_state, bbox, &fills));
    }

    #[test]
    fn text_visibility_returns_true_when_fill_luminance_unknown() {
        let text_state = make_visible_text_state((1.0, 1.0, 1.0));
        let bbox = make_rect(0.0, 0.0, 1.0, 1.0);
        let fills = vec![PaintedFill {
            bbox: make_rect(0.0, 0.0, 1.0, 1.0),
            bg_luminance: None,
        }];
        assert!(text_is_visible_at_bbox(&text_state, bbox, &fills));
    }

    #[test]
    fn content_tokenizer_skips_inline_image_data_and_resumes_text_ops() {
        let content =
            b"BI /W 2 /H 2 /BPC 1 /IM true ID \x89xEIy\x00\x01 EI BT /F1 12 Tf (Hello) Tj ET";
        let mut tokenizer = ContentTokenizer::new(content);
        let mut operands = Vec::new();
        let mut ops = Vec::new();
        while let Some(op) = tokenizer.next_op_into(&mut operands) {
            ops.push(op);
        }
        assert_eq!(ops, vec!["BT", "Tf", "Tj", "ET"]);
    }

    #[test]
    fn content_tokenizer_resumes_after_relaxed_ei_terminator() {
        let content = b"BI /W 1 /H 1 /BPC 1 /IM true ID \xffEI Q BT /F1 12 Tf (A) Tj ET";
        let mut tokenizer = ContentTokenizer::new(content);
        let mut operands = Vec::new();
        let mut ops = Vec::new();
        while let Some(op) = tokenizer.next_op_into(&mut operands) {
            ops.push(op);
        }
        assert_eq!(ops, vec!["Q", "BT", "Tf", "Tj", "ET"]);
    }

    #[test]
    fn decode_stream_resolves_indirect_flate_filter_reference() {
        let raw = b"BT /F1 12 Tf (INDIRECT FILTER OK) Tj ET\n";
        let mut encoder =
            flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::default());
        encoder.write_all(raw).expect("compress test stream");
        let compressed = encoder.finish().expect("finish compression");

        let mut objects = HashMap::new();
        objects.insert(
            (1, 0),
            Object::Stream {
                dict: HashMap::from([(
                    "Filter".to_string(),
                    Object::Reference {
                        obj_num: 2,
                        gen_num: 0,
                    },
                )]),
                data: compressed,
            },
        );
        objects.insert(
            (2, 0),
            Object::Array(vec![Object::Name("FlateDecode".to_string())]),
        );

        let doc = PdfDoc {
            objects,
            trailer: None,
        };
        let decoded = decode_stream(
            &doc,
            &Object::Reference {
                obj_num: 1,
                gen_num: 0,
            },
        )
        .expect("decoded stream");
        assert_eq!(decoded, raw);
    }
}
