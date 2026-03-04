use std::collections::BTreeMap;
use std::collections::HashMap;
use std::collections::VecDeque;
use std::sync::{Arc, Mutex, OnceLock};

use regex::RegexBuilder;

use crate::text::{CharBBox, Rectangle, TextBlock};

#[derive(Debug, Clone, Copy)]
pub struct RectF {
    pub min_x: f64,
    pub min_y: f64,
    pub max_x: f64,
    pub max_y: f64,
}

impl RectF {
    pub fn from_ltrb(left: f64, top: f64, right: f64, bottom: f64) -> Self {
        let min_x = left.min(right);
        let max_x = left.max(right);
        let min_y = bottom.min(top);
        let max_y = bottom.max(top);
        Self {
            min_x,
            min_y,
            max_x,
            max_y,
        }
    }

    pub fn intersects(&self, other: &RectF) -> bool {
        self.min_x <= other.max_x
            && self.max_x >= other.min_x
            && self.min_y <= other.max_y
            && self.max_y >= other.min_y
    }

    pub fn union(&self, other: &RectF) -> RectF {
        RectF {
            min_x: self.min_x.min(other.min_x),
            min_y: self.min_y.min(other.min_y),
            max_x: self.max_x.max(other.max_x),
            max_y: self.max_y.max(other.max_y),
        }
    }

    pub fn area(&self) -> f64 {
        let w = (self.max_x - self.min_x).abs();
        let h = (self.max_y - self.min_y).abs();
        w * h
    }

    pub fn intersection_area(&self, other: &RectF) -> f64 {
        let min_x = self.min_x.max(other.min_x);
        let min_y = self.min_y.max(other.min_y);
        let max_x = self.max_x.min(other.max_x);
        let max_y = self.max_y.min(other.max_y);
        let w = (max_x - min_x).max(0.0);
        let h = (max_y - min_y).max(0.0);
        w * h
    }
}

#[derive(Debug)]
enum Node {
    Leaf { bbox: RectF, items: Vec<usize> },
    Internal { bbox: RectF, children: Vec<usize> },
}

#[derive(Debug)]
pub struct RTree {
    nodes: Vec<Node>,
    root: usize,
}

impl RTree {
    pub fn build(rects: &[RectF], max_entries: usize) -> Self {
        let max_entries = max_entries.max(4);
        let mut indices: Vec<usize> = (0..rects.len()).collect();
        indices.sort_by(|&a, &b| rects[a].min_x.partial_cmp(&rects[b].min_x).unwrap());

        let mut nodes: Vec<Node> = Vec::new();
        let mut level: Vec<usize> = Vec::new();

        for chunk in indices.chunks(max_entries) {
            let mut bbox = rects[chunk[0]];
            for &idx in chunk.iter().skip(1) {
                bbox = bbox.union(&rects[idx]);
            }
            let node_idx = nodes.len();
            nodes.push(Node::Leaf {
                bbox,
                items: chunk.to_vec(),
            });
            level.push(node_idx);
        }

        while level.len() > 1 {
            let mut next_level: Vec<usize> = Vec::new();
            level.sort_by(|&a, &b| {
                let ba = node_bbox(&nodes[a]);
                let bb = node_bbox(&nodes[b]);
                ba.min_x.partial_cmp(&bb.min_x).unwrap()
            });
            for chunk in level.chunks(max_entries) {
                let mut bbox = node_bbox(&nodes[chunk[0]]);
                for &idx in chunk.iter().skip(1) {
                    bbox = bbox.union(&node_bbox(&nodes[idx]));
                }
                let node_idx = nodes.len();
                nodes.push(Node::Internal {
                    bbox,
                    children: chunk.to_vec(),
                });
                next_level.push(node_idx);
            }
            level = next_level;
        }

        let root = if level.is_empty() {
            let empty_bbox = RectF {
                min_x: 0.0,
                min_y: 0.0,
                max_x: 0.0,
                max_y: 0.0,
            };
            let node_idx = nodes.len();
            nodes.push(Node::Leaf {
                bbox: empty_bbox,
                items: Vec::new(),
            });
            node_idx
        } else {
            level[0]
        };

        Self { nodes, root }
    }

    pub fn search(&self, rects: &[RectF], query: &RectF, overlap: f64) -> Vec<usize> {
        let mut out: Vec<usize> = Vec::new();
        let mut stack: Vec<usize> = vec![self.root];
        let overlap = overlap.clamp(0.00001, 1.0);
        while let Some(idx) = stack.pop() {
            match &self.nodes[idx] {
                Node::Leaf { bbox, items } => {
                    if !bbox.intersects(query) {
                        continue;
                    }
                    for &item in items {
                        let r = &rects[item];
                        if !r.intersects(query) {
                            continue;
                        }
                        let area = r.area();
                        if area <= 0.0 {
                            continue;
                        }
                        let inter = r.intersection_area(query);
                        if inter >= (overlap * area) {
                            out.push(item);
                        }
                    }
                }
                Node::Internal { bbox, children } => {
                    if !bbox.intersects(query) {
                        continue;
                    }
                    for &child in children {
                        stack.push(child);
                    }
                }
            }
        }
        out
    }
}

fn node_bbox(node: &Node) -> RectF {
    match node {
        Node::Leaf { bbox, .. } => *bbox,
        Node::Internal { bbox, .. } => *bbox,
    }
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
struct CacheKey {
    left: u64,
    top: u64,
    right: u64,
    bottom: u64,
    overlap: u64,
}

impl CacheKey {
    fn from(left: f64, top: f64, right: f64, bottom: f64, overlap: f64) -> Self {
        Self {
            left: left.to_bits(),
            top: top.to_bits(),
            right: right.to_bits(),
            bottom: bottom.to_bits(),
            overlap: overlap.to_bits(),
        }
    }
}

#[derive(Debug)]
struct LruCache {
    capacity: usize,
    map: HashMap<CacheKey, Vec<usize>>,
    order: VecDeque<CacheKey>,
}

impl LruCache {
    fn new(capacity: usize) -> Self {
        Self {
            capacity: capacity.max(1),
            map: HashMap::new(),
            order: VecDeque::new(),
        }
    }

    fn get(&mut self, key: &CacheKey) -> Option<Vec<usize>> {
        self.map.get(key).cloned()
    }

    fn insert(&mut self, key: CacheKey, value: Vec<usize>) {
        if self.map.contains_key(&key) {
            self.map.insert(key, value);
            return;
        }
        self.map.insert(key, value);
        self.order.push_back(key);
        if self.order.len() > self.capacity {
            if let Some(old) = self.order.pop_front() {
                self.map.remove(&old);
            }
        }
    }
}

#[derive(Debug)]
pub struct TextBlockIndex {
    blocks: Vec<TextBlock>,
    block_chars: Option<Arc<Vec<Vec<CharBBox>>>>,
    rects: Vec<RectF>,
    rtree: RTree,
    cache: LruCache,
    regex_cache: RegexLruCache,
    regex_indices_cache: RegexIndicesLruCache,
    regex_index: OnceLock<RegexSearchIndex>,
    doc_rect: Rectangle,
    page_rects: Vec<Rectangle>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ScopedMergeRules {
    pub no_numeric_pair_merge: bool,
    pub no_date_pair_merge: bool,
}

#[derive(Debug)]
pub enum RegexSearchError {
    InvalidPattern(regex::Error),
}

#[derive(Debug)]
pub enum SplitError {
    InvalidPattern(regex::Error),
    MissingBlockChars,
}

#[derive(Debug)]
struct CompiledRegexCache {
    capacity: usize,
    map: HashMap<String, regex::Regex>,
    order: VecDeque<String>,
}

impl CompiledRegexCache {
    fn new(capacity: usize) -> Self {
        Self {
            capacity: capacity.max(1),
            map: HashMap::new(),
            order: VecDeque::new(),
        }
    }

    fn get_or_compile(&mut self, pattern: &str) -> Result<regex::Regex, regex::Error> {
        if let Some(regex) = self.map.get(pattern).cloned() {
            return Ok(regex);
        }

        let compiled = RegexBuilder::new(pattern).case_insensitive(true).build()?;
        self.map.insert(pattern.to_string(), compiled.clone());
        self.order.push_back(pattern.to_string());
        if self.order.len() > self.capacity
            && let Some(old) = self.order.pop_front()
        {
            self.map.remove(&old);
        }
        Ok(compiled)
    }
}

fn compiled_regex(pattern: &str) -> Result<regex::Regex, regex::Error> {
    static CACHE: OnceLock<Mutex<CompiledRegexCache>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(CompiledRegexCache::new(256)));
    let mut guard = match cache.lock() {
        Ok(g) => g,
        Err(poisoned) => poisoned.into_inner(),
    };
    guard.get_or_compile(pattern)
}

#[derive(Debug)]
struct RegexLruCache {
    capacity: usize,
    map: HashMap<String, Vec<TextBlock>>,
    order: VecDeque<String>,
}

impl RegexLruCache {
    fn new(capacity: usize) -> Self {
        Self {
            capacity: capacity.max(1),
            map: HashMap::new(),
            order: VecDeque::new(),
        }
    }

    fn get(&mut self, key: &str) -> Option<Vec<TextBlock>> {
        self.map.get(key).cloned()
    }

    fn insert(&mut self, key: String, value: Vec<TextBlock>) {
        if self.map.contains_key(&key) {
            self.map.insert(key, value);
            return;
        }
        self.map.insert(key.clone(), value);
        self.order.push_back(key);
        if self.order.len() > self.capacity {
            if let Some(old) = self.order.pop_front() {
                self.map.remove(&old);
            }
        }
    }
}

#[derive(Debug)]
struct RegexIndicesLruCache {
    capacity: usize,
    map: HashMap<String, Vec<Vec<usize>>>,
    order: VecDeque<String>,
}

impl RegexIndicesLruCache {
    fn new(capacity: usize) -> Self {
        Self {
            capacity: capacity.max(1),
            map: HashMap::new(),
            order: VecDeque::new(),
        }
    }

    fn get(&mut self, key: &str) -> Option<Vec<Vec<usize>>> {
        self.map.get(key).cloned()
    }

    fn insert(&mut self, key: String, value: Vec<Vec<usize>>) {
        if self.map.contains_key(&key) {
            self.map.insert(key, value);
            return;
        }
        self.map.insert(key.clone(), value);
        self.order.push_back(key);
        if self.order.len() > self.capacity {
            if let Some(old) = self.order.pop_front() {
                self.map.remove(&old);
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct WordSpan {
    start: usize,
    end: usize,
    block_index: usize,
}

#[derive(Debug)]
struct RegexPageIndex {
    body: String,
    spans: Vec<WordSpan>,
}

#[derive(Debug)]
struct RegexSearchIndex {
    pages: Vec<RegexPageIndex>,
}

const REGEX_LINE_VERTICAL_OVERLAP_THRESHOLD: f64 = 0.8;
const REGEX_LINE_HEIGHT_RATIO_LIMIT: f64 = 1.5;
const MAX_DENSE_INFERRED_PAGE_RECTS: usize = 1_000_000;

impl TextBlockIndex {
    pub fn new(blocks: Vec<TextBlock>) -> Self {
        Self::from_parts(blocks, None, None)
    }

    pub fn new_with_chars(blocks: Vec<TextBlock>, block_chars: Vec<Vec<CharBBox>>) -> Self {
        assert_eq!(
            blocks.len(),
            block_chars.len(),
            "block_chars must align one-to-one with blocks"
        );
        Self::from_parts(blocks, Some(Arc::new(block_chars)), None)
    }

    pub fn new_with_chars_and_page_rects(
        blocks: Vec<TextBlock>,
        block_chars: Vec<Vec<CharBBox>>,
        page_rects: Vec<Rectangle>,
    ) -> Self {
        assert_eq!(
            blocks.len(),
            block_chars.len(),
            "block_chars must align one-to-one with blocks"
        );
        Self::from_parts(blocks, Some(Arc::new(block_chars)), Some(page_rects))
    }

    fn from_parts(
        blocks: Vec<TextBlock>,
        block_chars: Option<Arc<Vec<Vec<CharBBox>>>>,
        page_rects: Option<Vec<Rectangle>>,
    ) -> Self {
        let rects: Vec<RectF> = blocks
            .iter()
            .map(|b| RectF {
                min_x: b.bbox.left,
                min_y: b.bbox.top,
                max_x: b.bbox.right,
                max_y: b.bbox.bottom,
            })
            .collect();
        let page_rects = page_rects
            .map(|rects| rects.into_iter().map(normalize_rectangle).collect())
            .unwrap_or_else(|| infer_page_rects_from_blocks(&blocks));
        let doc_rect = doc_rect_from_page_rects(&page_rects);
        let rtree = RTree::build(&rects, 16);
        Self {
            blocks,
            block_chars,
            rects,
            rtree,
            cache: LruCache::new(1024),
            regex_cache: RegexLruCache::new(128),
            regex_indices_cache: RegexIndicesLruCache::new(128),
            regex_index: OnceLock::new(),
            doc_rect,
            page_rects,
        }
    }

    pub fn search(&mut self, rect: Rectangle, overlap: f64) -> Vec<TextBlock> {
        let indices = self.search_indices(rect, overlap);
        indices
            .into_iter()
            .filter_map(|i| self.blocks.get(i).cloned())
            .collect()
    }

    pub fn search_indices(&mut self, rect: Rectangle, overlap: f64) -> Vec<usize> {
        let overlap = overlap.clamp(0.0, 1.0);
        let key = CacheKey::from(rect.left, rect.top, rect.right, rect.bottom, overlap);
        if let Some(indices) = self.cache.get(&key) {
            return indices;
        }
        let indices = self.matching_block_indices(rect, overlap);
        self.cache.insert(key, indices.clone());
        indices
    }

    pub fn scoped(
        &self,
        rect: Rectangle,
        overlap: f64,
        merge_threshold: f64,
        normalize: bool,
        rules: ScopedMergeRules,
    ) -> Self {
        let mut indices = self.matching_block_indices(rect, overlap);
        indices.sort_unstable();
        let mut scoped_blocks: Vec<ScopedBlock> = indices
            .into_iter()
            .map(|index| ScopedBlock {
                block: self.blocks[index].clone(),
                chars: self
                    .block_chars
                    .as_ref()
                    .and_then(|all| all.get(index))
                    .cloned()
                    .unwrap_or_default(),
            })
            .collect();
        if normalize {
            scoped_blocks = normalize_scoped_blocks(scoped_blocks);
        }
        if merge_threshold > 0.0 {
            scoped_blocks = merge_scoped_blocks(scoped_blocks, merge_threshold, rules);
        }
        let mut blocks = Vec::with_capacity(scoped_blocks.len());
        let mut block_chars = Vec::new();
        let keep_chars = self.block_chars.is_some();
        if keep_chars {
            block_chars.reserve(scoped_blocks.len());
        }
        for scoped in scoped_blocks {
            blocks.push(scoped.block);
            if keep_chars {
                block_chars.push(scoped.chars);
            }
        }
        let page_rects = Some(self.page_rects.clone());
        if keep_chars {
            Self::from_parts(blocks, Some(Arc::new(block_chars)), page_rects)
        } else {
            Self::from_parts(blocks, None, page_rects)
        }
    }

    pub fn blocks(&self) -> Vec<TextBlock> {
        self.blocks.clone()
    }

    pub fn block_at(&self, index: usize) -> Option<&TextBlock> {
        self.blocks.get(index)
    }

    pub fn block_chars_at(&self, index: usize) -> Option<&[CharBBox]> {
        self.block_chars
            .as_ref()
            .and_then(|all| all.get(index))
            .map(Vec::as_slice)
    }

    pub fn block_chars_arc(&self) -> Option<Arc<Vec<Vec<CharBBox>>>> {
        self.block_chars.as_ref().map(Arc::clone)
    }

    pub fn block_len(&self) -> usize {
        self.blocks.len()
    }

    pub fn text(&self) -> String {
        let regex_index = self.ensure_regex_index();
        let mut out = String::new();
        for page in &regex_index.pages {
            out.push_str(&page.body);
            out.push('\n');
        }
        out
    }

    pub fn search_regex(&mut self, pattern: &str) -> Result<Vec<TextBlock>, RegexSearchError> {
        if let Some(cached) = self.regex_cache.get(pattern) {
            return Ok(cached);
        }

        let indices = self.search_regex_indices(pattern)?;
        let mut out: Vec<TextBlock> = Vec::with_capacity(indices.len());
        for block_indices in indices {
            if let Some(block) = text_block_from_blocks(&self.blocks, &block_indices) {
                out.push(block);
            }
        }
        self.regex_cache.insert(pattern.to_string(), out.clone());
        Ok(out)
    }

    pub fn search_regex_exists(&mut self, pattern: &str) -> Result<bool, RegexSearchError> {
        if let Some(cached) = self.regex_indices_cache.get(pattern) {
            return Ok(!cached.is_empty());
        }

        let regex_index = self.ensure_regex_index();
        let regex = compiled_regex(pattern).map_err(RegexSearchError::InvalidPattern)?;
        for page in &regex_index.pages {
            if regex.is_match(&page.body) {
                return Ok(true);
            }
        }

        // Cache misses to avoid rescanning the document for repeated existence checks.
        self.regex_indices_cache
            .insert(pattern.to_string(), Vec::new());
        Ok(false)
    }

    pub fn search_regex_indices(
        &mut self,
        pattern: &str,
    ) -> Result<Vec<Vec<usize>>, RegexSearchError> {
        if let Some(cached) = self.regex_indices_cache.get(pattern) {
            return Ok(cached);
        }

        let regex_index = self.ensure_regex_index();
        let regex = compiled_regex(pattern).map_err(RegexSearchError::InvalidPattern)?;
        let mut out: Vec<Vec<usize>> = Vec::new();
        for page in &regex_index.pages {
            for mat in regex.find_iter(&page.body) {
                if mat.start() >= mat.end() {
                    continue;
                }
                let block_indices = block_indices_for_match(&page.spans, mat.start(), mat.end());
                if block_indices.is_empty() {
                    continue;
                }
                out.push(block_indices);
            }
        }
        self.regex_indices_cache
            .insert(pattern.to_string(), out.clone());
        Ok(out)
    }

    pub fn split(&self, pattern: &str) -> Result<Self, SplitError> {
        let regex = compiled_regex(pattern).map_err(SplitError::InvalidPattern)?;
        let Some(block_chars) = self.block_chars.as_ref() else {
            return Err(SplitError::MissingBlockChars);
        };

        let mut out_blocks: Option<Vec<TextBlock>> = None;
        let mut out_block_chars: Option<Vec<Vec<CharBBox>>> = None;

        for (index, block) in self.blocks.iter().enumerate() {
            let chars = block_chars
                .get(index)
                .map(Vec::as_slice)
                .unwrap_or_default();
            let maybe_match = regex
                .find_iter(&block.text)
                .find(|mat| mat.start() < mat.end());
            let Some(mat) = maybe_match else {
                if let (Some(blocks_out), Some(chars_out)) =
                    (out_blocks.as_mut(), out_block_chars.as_mut())
                {
                    blocks_out.push(block.clone());
                    chars_out.push(chars.to_vec());
                }
                continue;
            };

            let Some(split_blocks) = split_block_by_match(block, chars, mat.start(), mat.end())
            else {
                if let (Some(blocks_out), Some(chars_out)) =
                    (out_blocks.as_mut(), out_block_chars.as_mut())
                {
                    blocks_out.push(block.clone());
                    chars_out.push(chars.to_vec());
                }
                continue;
            };
            let changed = split_blocks.len() > 1
                || split_blocks
                    .first()
                    .is_some_and(|(split, _)| split.text != block.text || split.bbox != block.bbox);
            if !changed {
                if let (Some(blocks_out), Some(chars_out)) =
                    (out_blocks.as_mut(), out_block_chars.as_mut())
                {
                    blocks_out.push(block.clone());
                    chars_out.push(chars.to_vec());
                }
                continue;
            }

            if out_blocks.is_none() {
                let mut blocks_out = Vec::with_capacity(self.blocks.len().saturating_add(2));
                let mut chars_out = Vec::with_capacity(self.blocks.len().saturating_add(2));
                for prior in 0..index {
                    blocks_out.push(self.blocks[prior].clone());
                    chars_out.push(block_chars[prior].clone());
                }
                out_blocks = Some(blocks_out);
                out_block_chars = Some(chars_out);
            }

            if let (Some(blocks_out), Some(chars_out)) =
                (out_blocks.as_mut(), out_block_chars.as_mut())
            {
                for (split, split_chars) in split_blocks {
                    blocks_out.push(split);
                    chars_out.push(split_chars);
                }
            }
        }

        if let (Some(blocks_out), Some(chars_out)) = (out_blocks, out_block_chars) {
            return Ok(Self::from_parts(
                blocks_out,
                Some(Arc::new(chars_out)),
                Some(self.page_rects.clone()),
            ));
        }

        Ok(Self::from_parts(
            self.blocks.clone(),
            Some(Arc::clone(block_chars)),
            Some(self.page_rects.clone()),
        ))
    }

    pub fn doc_rect(&self) -> Rectangle {
        self.doc_rect
    }

    pub fn page_rects(&self) -> Vec<Rectangle> {
        self.page_rects.clone()
    }

    pub fn page_rects_slice(&self) -> &[Rectangle] {
        &self.page_rects
    }

    fn matching_block_indices(&self, rect: Rectangle, overlap: f64) -> Vec<usize> {
        let overlap = overlap.clamp(0.0, 1.0);
        let query = RectF::from_ltrb(rect.left, rect.top, rect.right, rect.bottom);
        self.rtree.search(&self.rects, &query, overlap)
    }

    fn ensure_regex_index(&self) -> &RegexSearchIndex {
        self.regex_index
            .get_or_init(|| build_regex_index(&self.blocks))
    }
}

fn doc_rect_from_page_rects(page_rects: &[Rectangle]) -> Rectangle {
    let mut iter = page_rects
        .iter()
        .copied()
        .map(normalize_rectangle)
        .filter(|r| {
            r.left.is_finite()
                && r.top.is_finite()
                && r.right.is_finite()
                && r.bottom.is_finite()
                && r.left < r.right
                && r.top < r.bottom
        });
    let Some(first) = iter.next() else {
        return Rectangle {
            top: 0.0,
            left: 0.0,
            bottom: 0.0,
            right: 0.0,
        };
    };
    let mut min_x = first.left;
    let mut min_y = first.top;
    let mut max_x = first.right;
    let mut max_y = first.bottom;
    for r in iter {
        min_x = min_x.min(r.left);
        min_y = min_y.min(r.top);
        max_x = max_x.max(r.right);
        max_y = max_y.max(r.bottom);
    }
    Rectangle {
        top: min_y,
        left: min_x,
        bottom: max_y,
        right: max_x,
    }
}

fn infer_page_rects_from_blocks(blocks: &[TextBlock]) -> Vec<Rectangle> {
    if blocks.is_empty() {
        return Vec::new();
    }
    let mut max_page_index = 0usize;
    let mut per_page: BTreeMap<usize, Rectangle> = BTreeMap::new();
    for block in blocks {
        let rect = normalize_rectangle(block.bbox);
        max_page_index = max_page_index.max(block.page_index);
        per_page
            .entry(block.page_index)
            .and_modify(|existing| *existing = union_rectangles(*existing, rect))
            .or_insert(rect);
    }
    let Some(dense_len) = max_page_index.checked_add(1) else {
        return per_page.into_values().collect();
    };
    if dense_len > MAX_DENSE_INFERRED_PAGE_RECTS {
        return per_page.into_values().collect();
    }
    let mut page_rects = vec![empty_rectangle(); dense_len];
    for (page_index, rect) in per_page {
        page_rects[page_index] = rect;
    }
    page_rects
}

fn empty_rectangle() -> Rectangle {
    Rectangle {
        top: 0.0,
        left: 0.0,
        bottom: 0.0,
        right: 0.0,
    }
}

fn normalize_rectangle(rect: Rectangle) -> Rectangle {
    Rectangle {
        top: rect.top.min(rect.bottom),
        left: rect.left.min(rect.right),
        bottom: rect.top.max(rect.bottom),
        right: rect.left.max(rect.right),
    }
}

#[derive(Debug, Clone)]
struct ScopedBlock {
    block: TextBlock,
    chars: Vec<CharBBox>,
}

fn normalize_scoped_blocks(blocks: Vec<ScopedBlock>) -> Vec<ScopedBlock> {
    if blocks.len() < 2 {
        return blocks;
    }

    let mut per_page: HashMap<usize, Vec<ScopedBlock>> = HashMap::new();
    for block in blocks {
        per_page
            .entry(block.block.page_index)
            .or_default()
            .push(block);
    }

    let mut page_ids: Vec<usize> = per_page.keys().copied().collect();
    page_ids.sort_unstable();

    let mut normalized: Vec<ScopedBlock> = Vec::new();
    for page_id in page_ids {
        let page_blocks = per_page.remove(&page_id).unwrap_or_default();
        normalized.extend(normalize_page_blocks(page_blocks));
    }

    sort_blocks_for_reading_order(&mut normalized);
    normalized
}

fn normalize_page_blocks(mut blocks: Vec<ScopedBlock>) -> Vec<ScopedBlock> {
    if blocks.len() < 2 {
        return blocks;
    }

    blocks.sort_by(|a, b| {
        a.block
            .bbox
            .top
            .total_cmp(&b.block.bbox.top)
            .then_with(|| a.block.bbox.left.total_cmp(&b.block.bbox.left))
            .then_with(|| a.block.bbox.right.total_cmp(&b.block.bbox.right))
    });

    let mut normalized: Vec<ScopedBlock> = Vec::with_capacity(blocks.len());
    let mut current_line: Vec<ScopedBlock> = Vec::new();

    for block in blocks {
        if current_line.is_empty() {
            current_line.push(block);
            continue;
        }

        let anchor = &current_line[0];
        let same_line = anchor.block.bbox.vertical_overlap(&block.block.bbox, 0.5)
            || block.block.bbox.vertical_overlap(&anchor.block.bbox, 0.5);
        if same_line {
            current_line.push(block);
            continue;
        }

        normalized.extend(normalize_line_blocks(current_line));
        current_line = vec![block];
    }

    if !current_line.is_empty() {
        normalized.extend(normalize_line_blocks(current_line));
    }

    normalized
}

fn normalize_line_blocks(mut line: Vec<ScopedBlock>) -> Vec<ScopedBlock> {
    if line.len() < 2 {
        return line;
    }

    let mut top_sum = 0.0;
    let mut bottom_sum = 0.0;
    for block in &line {
        top_sum += block.block.bbox.top.min(block.block.bbox.bottom);
        bottom_sum += block.block.bbox.top.max(block.block.bbox.bottom);
    }
    let n = line.len() as f64;
    let avg_top = top_sum / n;
    let avg_bottom = bottom_sum / n;

    for block in &mut line {
        block.block.bbox.top = avg_top;
        block.block.bbox.bottom = avg_bottom;
    }

    line
}

fn merge_scoped_blocks(
    blocks: Vec<ScopedBlock>,
    merge_threshold: f64,
    rules: ScopedMergeRules,
) -> Vec<ScopedBlock> {
    if blocks.len() < 2 {
        return blocks;
    }

    let merge_threshold = merge_threshold.max(0.0);
    if merge_threshold <= 0.0 {
        return blocks;
    }

    let mut per_page: HashMap<usize, Vec<ScopedBlock>> = HashMap::new();
    for block in blocks {
        per_page
            .entry(block.block.page_index)
            .or_default()
            .push(block);
    }

    let mut page_ids: Vec<usize> = per_page.keys().copied().collect();
    page_ids.sort_unstable();

    let mut merged: Vec<ScopedBlock> = Vec::new();
    for page_id in page_ids {
        let page_blocks = per_page.remove(&page_id).unwrap_or_default();
        merged.extend(merge_page_blocks(page_blocks, merge_threshold, rules));
    }

    sort_blocks_for_reading_order(&mut merged);
    merged
}

fn sort_blocks_for_reading_order(blocks: &mut [ScopedBlock]) {
    blocks.sort_by(|a, b| {
        a.block
            .page_index
            .cmp(&b.block.page_index)
            .then_with(|| a.block.bbox.top.total_cmp(&b.block.bbox.top))
            .then_with(|| a.block.bbox.left.total_cmp(&b.block.bbox.left))
            .then_with(|| a.block.bbox.right.total_cmp(&b.block.bbox.right))
    });
}

#[derive(Debug)]
struct DisjointSet {
    parent: Vec<usize>,
    rank: Vec<u8>,
}

impl DisjointSet {
    fn new(size: usize) -> Self {
        Self {
            parent: (0..size).collect(),
            rank: vec![0; size],
        }
    }

    fn find(&mut self, node: usize) -> usize {
        if self.parent[node] != node {
            let root = self.find(self.parent[node]);
            self.parent[node] = root;
        }
        self.parent[node]
    }

    fn union(&mut self, a: usize, b: usize) {
        let root_a = self.find(a);
        let root_b = self.find(b);
        if root_a == root_b {
            return;
        }
        if self.rank[root_a] < self.rank[root_b] {
            self.parent[root_a] = root_b;
            return;
        }
        if self.rank[root_a] > self.rank[root_b] {
            self.parent[root_b] = root_a;
            return;
        }
        self.parent[root_b] = root_a;
        self.rank[root_a] = self.rank[root_a].saturating_add(1);
    }
}

fn merge_page_blocks(
    blocks: Vec<ScopedBlock>,
    merge_threshold: f64,
    rules: ScopedMergeRules,
) -> Vec<ScopedBlock> {
    if blocks.len() < 2 {
        return blocks;
    }

    let classes: Vec<BlockClass> = blocks
        .iter()
        .map(|block| classify_block_text(&block.block.text))
        .collect();
    let spans: Vec<(f64, f64)> = blocks
        .iter()
        .map(|block| {
            (
                block.block.bbox.left.min(block.block.bbox.right),
                block.block.bbox.left.max(block.block.bbox.right),
            )
        })
        .collect();
    let mut order: Vec<usize> = (0..blocks.len()).collect();
    order.sort_by(|&a, &b| {
        spans[a]
            .0
            .total_cmp(&spans[b].0)
            .then_with(|| spans[a].1.total_cmp(&spans[b].1))
            .then_with(|| {
                blocks[a]
                    .block
                    .bbox
                    .top
                    .total_cmp(&blocks[b].block.bbox.top)
            })
            .then_with(|| a.cmp(&b))
    });

    let mut dsu = DisjointSet::new(blocks.len());
    let mut active: Vec<usize> = Vec::new();
    for curr in order {
        let curr_left = spans[curr].0;
        active.retain(|&candidate| spans[candidate].1 + merge_threshold >= curr_left);
        for &candidate in &active {
            if blocks_are_mergeable(
                &blocks[curr].block,
                classes[curr],
                &blocks[candidate].block,
                classes[candidate],
                merge_threshold,
                rules,
            ) {
                dsu.union(curr, candidate);
            }
        }
        active.push(curr);
    }

    let mut grouped: HashMap<usize, Vec<usize>> = HashMap::new();
    for idx in 0..blocks.len() {
        let root = dsu.find(idx);
        grouped.entry(root).or_default().push(idx);
    }
    let mut components: Vec<Vec<usize>> = grouped.into_values().collect();
    components.sort_by(|a, b| {
        let a_min = a.iter().copied().min().unwrap_or(usize::MAX);
        let b_min = b.iter().copied().min().unwrap_or(usize::MAX);
        a_min.cmp(&b_min)
    });

    let mut merged: Vec<ScopedBlock> = Vec::with_capacity(components.len());
    for component in components {
        merged.push(merge_component(&blocks, &component));
    }
    merged
}

fn blocks_are_mergeable(
    a: &TextBlock,
    a_class: BlockClass,
    b: &TextBlock,
    b_class: BlockClass,
    merge_threshold: f64,
    rules: ScopedMergeRules,
) -> bool {
    if a.page_index != b.page_index {
        return false;
    }
    let same_line = a.bbox.vertical_overlap(&b.bbox, 0.5) || b.bbox.vertical_overlap(&a.bbox, 0.5);
    if !same_line {
        return false;
    }
    if rules.no_numeric_pair_merge && a_class.is_numeric() && b_class.is_numeric() {
        return false;
    }
    if rules.no_date_pair_merge && a_class.is_date() && b_class.is_date() {
        return false;
    }
    horizontal_gap(a.bbox, b.bbox) <= merge_threshold
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct BlockClass(u8);

impl BlockClass {
    const NONE: Self = Self(0);
    const NUMERIC: Self = Self(1);
    const DATE: Self = Self(1 << 1);

    fn with_numeric(mut self) -> Self {
        self.0 |= Self::NUMERIC.0;
        self
    }

    fn with_date(mut self) -> Self {
        self.0 |= Self::DATE.0;
        self
    }

    fn is_numeric(self) -> bool {
        self.0 & Self::NUMERIC.0 != 0
    }

    fn is_date(self) -> bool {
        self.0 & Self::DATE.0 != 0
    }
}

fn classify_block_text(text: &str) -> BlockClass {
    let text = text.trim();
    if text.is_empty() {
        return BlockClass::NONE;
    }
    let mut class = BlockClass::NONE;
    if is_numeric_token(text) {
        class = class.with_numeric();
    }
    if is_date_token(text) {
        class = class.with_date();
    }
    class
}

fn is_numeric_token(text: &str) -> bool {
    let text = text.trim();
    if text.is_empty() {
        return false;
    }

    let bytes = text.as_bytes();
    let len = bytes.len();
    let mut idx = 0usize;

    if bytes[idx] == b'(' {
        idx += 1;
        if idx >= len {
            return false;
        }
    }

    let mut saw_prefix_dollar = false;
    let mut saw_prefix_minus = false;
    loop {
        if idx >= len {
            return false;
        }
        match bytes[idx] {
            b'$' if !saw_prefix_dollar => {
                saw_prefix_dollar = true;
                idx += 1;
            }
            b'-' if !saw_prefix_minus => {
                saw_prefix_minus = true;
                idx += 1;
            }
            _ => break,
        }
    }

    let mut digit_count = 0usize;
    if idx < len && bytes[idx].is_ascii_digit() {
        let int_start = idx;
        while idx < len && bytes[idx].is_ascii_digit() {
            idx += 1;
        }
        let first_group_digits = idx - int_start;
        digit_count += first_group_digits;

        if idx < len && bytes[idx] == b',' {
            if first_group_digits > 3 {
                return false;
            }
            while idx < len && bytes[idx] == b',' {
                idx += 1;
                if idx + 3 > len {
                    return false;
                }
                for _ in 0..3 {
                    if !bytes[idx].is_ascii_digit() {
                        return false;
                    }
                    idx += 1;
                }
                digit_count += 3;
            }
            if idx < len && bytes[idx].is_ascii_digit() {
                return false;
            }
        }
    }

    if idx < len && bytes[idx] == b'.' {
        idx += 1;
        while idx < len && bytes[idx].is_ascii_digit() {
            idx += 1;
            digit_count += 1;
        }
    }

    if digit_count == 0 {
        return false;
    }

    if idx < len && bytes[idx] == b')' {
        idx += 1;
    }

    if idx < len && bytes[idx] == b'-' {
        idx += 1;
    }

    idx == len
}

fn is_date_token(text: &str) -> bool {
    let mut text = text.trim();
    if text.is_empty() {
        return false;
    }
    if let Some(stripped) = text.strip_prefix('(') {
        text = stripped;
    }
    if let Some(stripped) = text.strip_suffix(')') {
        text = stripped;
    }
    if text.is_empty() {
        return false;
    }

    let bytes = text.as_bytes();
    let mut idx = match parse_date_component(bytes, 0) {
        Some(end) => end,
        None => return false,
    };
    if idx == bytes.len() {
        return true;
    }
    if bytes[idx] != b'-' {
        return false;
    }
    idx += 1;
    match parse_date_component(bytes, idx) {
        Some(end) => end == bytes.len(),
        None => false,
    }
}

fn parse_date_component(bytes: &[u8], start: usize) -> Option<usize> {
    let mut idx = parse_digit_span(bytes, start, 1, 2)?;
    if idx >= bytes.len() || (bytes[idx] != b'/' && bytes[idx] != b'-') {
        return None;
    }
    idx += 1;

    idx = parse_digit_span(bytes, idx, 1, 2)?;
    if idx >= bytes.len() || (bytes[idx] != b'/' && bytes[idx] != b'-') {
        return None;
    }
    idx += 1;

    parse_digit_span(bytes, idx, 2, 4)
}

fn parse_digit_span(bytes: &[u8], start: usize, min_len: usize, max_len: usize) -> Option<usize> {
    let mut idx = start;
    let mut count = 0usize;
    while idx < bytes.len() && count < max_len && bytes[idx].is_ascii_digit() {
        idx += 1;
        count += 1;
    }
    if count < min_len {
        return None;
    }
    if idx < bytes.len() && bytes[idx].is_ascii_digit() {
        return None;
    }
    Some(idx)
}

fn horizontal_gap(a: Rectangle, b: Rectangle) -> f64 {
    let a_left = a.left.min(a.right);
    let a_right = a.left.max(a.right);
    let b_left = b.left.min(b.right);
    let b_right = b.left.max(b.right);

    if a_right >= b_left && b_right >= a_left {
        return 0.0;
    }
    if a_right < b_left {
        return b_left - a_right;
    }
    a_left - b_right
}

fn merge_component(blocks: &[ScopedBlock], component: &[usize]) -> ScopedBlock {
    let mut members: Vec<&ScopedBlock> = component.iter().map(|&index| &blocks[index]).collect();
    members.sort_by(|a, b| {
        a.block
            .bbox
            .left
            .total_cmp(&b.block.bbox.left)
            .then_with(|| a.block.bbox.top.total_cmp(&b.block.bbox.top))
            .then_with(|| a.block.bbox.right.total_cmp(&b.block.bbox.right))
    });

    let first = members[0];
    let mut text = String::new();
    let mut bbox = first.block.bbox;
    let mut chars = Vec::new();
    for (idx, block) in members.iter().enumerate() {
        if idx > 0 {
            text.push(' ');
        }
        text.push_str(&block.block.text);
        bbox = union_rectangles(bbox, block.block.bbox);
        chars.extend(block.chars.iter().cloned());
    }
    ScopedBlock {
        block: TextBlock {
            page_index: first.block.page_index,
            text,
            bbox,
        },
        chars,
    }
}

fn build_regex_index(blocks: &[TextBlock]) -> RegexSearchIndex {
    let mut per_page: HashMap<usize, Vec<usize>> = HashMap::new();
    for (idx, block) in blocks.iter().enumerate() {
        per_page.entry(block.page_index).or_default().push(idx);
    }

    let mut page_ids: Vec<usize> = per_page.keys().copied().collect();
    page_ids.sort_unstable();

    let mut pages: Vec<RegexPageIndex> = Vec::with_capacity(page_ids.len());
    for page_id in page_ids {
        let block_indices = per_page.remove(&page_id).unwrap_or_default();
        let lines = cluster_block_indices_into_lines(blocks, &block_indices);
        let (body, spans) = build_page_body(blocks, &lines);
        pages.push(RegexPageIndex { body, spans });
    }

    RegexSearchIndex { pages }
}

fn cluster_block_indices_into_lines(
    blocks: &[TextBlock],
    page_block_indices: &[usize],
) -> Vec<Vec<usize>> {
    if page_block_indices.is_empty() {
        return Vec::new();
    }

    #[derive(Clone, Copy)]
    struct LineEntry {
        index: usize,
        top: f64,
        bottom: f64,
        left: f64,
        right: f64,
        height: f64,
    }

    #[derive(Debug)]
    struct LineState {
        anchor_top: f64,
        anchor_bottom: f64,
        anchor_height: f64,
        line_top: f64,
        line_left: f64,
        indices: Vec<usize>,
    }

    let mut entries: Vec<LineEntry> = Vec::with_capacity(page_block_indices.len());
    for &global_index in page_block_indices {
        let block = &blocks[global_index];
        let left = block.bbox.left.min(block.bbox.right);
        let right = block.bbox.left.max(block.bbox.right);
        let top = block.bbox.top.min(block.bbox.bottom);
        let bottom = block.bbox.top.max(block.bbox.bottom);
        let height = (bottom - top).abs();
        entries.push(LineEntry {
            index: global_index,
            top,
            bottom,
            left,
            right,
            height,
        });
    }

    entries.sort_by(|a, b| {
        a.top
            .total_cmp(&b.top)
            .then_with(|| a.left.total_cmp(&b.left))
            .then_with(|| a.right.total_cmp(&b.right))
            .then_with(|| a.index.cmp(&b.index))
    });

    let mut lines: Vec<LineState> = Vec::new();
    for entry in entries {
        let mut best_line: Option<(usize, f64, f64)> = None;
        for (line_idx, line) in lines.iter().enumerate() {
            let overlap_top = line.anchor_top.max(entry.top);
            let overlap_bottom = line.anchor_bottom.min(entry.bottom);
            let overlap = (overlap_bottom - overlap_top).max(0.0);
            if overlap <= 0.0 || entry.height <= 0.0 {
                continue;
            }

            let overlap_ratio = overlap / entry.height;
            if overlap_ratio < REGEX_LINE_VERTICAL_OVERLAP_THRESHOLD {
                continue;
            }
            if line.anchor_height > (entry.height * REGEX_LINE_HEIGHT_RATIO_LIMIT) {
                continue;
            }

            let top_distance = (line.anchor_top - entry.top).abs();
            match best_line {
                Some((_, best_ratio, best_distance)) => {
                    let better_ratio = overlap_ratio > best_ratio;
                    let better_distance =
                        overlap_ratio == best_ratio && top_distance < best_distance;
                    if better_ratio || better_distance {
                        best_line = Some((line_idx, overlap_ratio, top_distance));
                    }
                }
                None => best_line = Some((line_idx, overlap_ratio, top_distance)),
            }
        }

        if let Some((line_idx, _, _)) = best_line {
            let line = &mut lines[line_idx];
            line.indices.push(entry.index);
            line.line_top = line.line_top.min(entry.top);
            line.line_left = line.line_left.min(entry.left);
        } else {
            lines.push(LineState {
                anchor_top: entry.top,
                anchor_bottom: entry.bottom,
                anchor_height: entry.height,
                line_top: entry.top,
                line_left: entry.left,
                indices: vec![entry.index],
            });
        }
    }

    for line in &mut lines {
        line.indices.sort_by(|&a, &b| {
            let ba = &blocks[a];
            let bb = &blocks[b];
            ba.bbox
                .left
                .total_cmp(&bb.bbox.left)
                .then_with(|| ba.bbox.top.total_cmp(&bb.bbox.top))
                .then_with(|| ba.bbox.right.total_cmp(&bb.bbox.right))
                .then_with(|| a.cmp(&b))
        });
    }
    lines.sort_by(|a, b| {
        a.line_top
            .total_cmp(&b.line_top)
            .then_with(|| a.line_left.total_cmp(&b.line_left))
    });
    lines.into_iter().map(|line| line.indices).collect()
}

fn build_page_body(blocks: &[TextBlock], lines: &[Vec<usize>]) -> (String, Vec<WordSpan>) {
    let mut body = String::new();
    let mut spans: Vec<WordSpan> = Vec::new();
    for (line_idx, line) in lines.iter().enumerate() {
        if line_idx > 0 {
            body.push('\n');
        }
        for (block_idx, &index) in line.iter().enumerate() {
            if block_idx > 0 {
                body.push(' ');
            }
            let start = body.len();
            body.push_str(&blocks[index].text);
            let end = body.len();
            if start < end {
                spans.push(WordSpan {
                    start,
                    end,
                    block_index: index,
                });
            }
        }
    }
    (body, spans)
}

fn block_indices_for_match(spans: &[WordSpan], start: usize, end: usize) -> Vec<usize> {
    if start >= end || spans.is_empty() {
        return Vec::new();
    }

    let mut indices: Vec<usize> = Vec::new();
    let first = spans.partition_point(|span| span.end <= start);
    for span in spans.iter().skip(first) {
        if span.start >= end {
            break;
        }
        if span.end > start && span.start < end {
            indices.push(span.block_index);
        }
    }
    indices
}

fn text_block_from_blocks(blocks: &[TextBlock], block_indices: &[usize]) -> Option<TextBlock> {
    let first = blocks.get(*block_indices.first()?)?;
    let mut text = first.text.clone();
    let mut bbox = first.bbox;
    for &index in block_indices.iter().skip(1) {
        let block = blocks.get(index)?;
        text.push(' ');
        text.push_str(&block.text);
        bbox = union_rectangles(bbox, block.bbox);
    }
    Some(TextBlock {
        page_index: first.page_index,
        text,
        bbox,
    })
}

fn split_block_by_match(
    block: &TextBlock,
    chars: &[CharBBox],
    match_start: usize,
    match_end: usize,
) -> Option<Vec<(TextBlock, Vec<CharBBox>)>> {
    if match_start >= match_end || match_end > block.text.len() {
        return None;
    }

    let (match_start_char, match_end_char) =
        byte_range_to_char_range(&block.text, match_start, match_end)?;
    let char_boundaries = map_text_char_boundaries_to_char_offsets(&block.text, chars)?;
    if match_end_char > char_boundaries.len().saturating_sub(1) {
        return None;
    }

    let left_char_end = char_boundaries[match_start_char];
    let match_char_start = char_boundaries[match_start_char];
    let match_char_end = char_boundaries[match_end_char];
    let right_char_start = char_boundaries[match_end_char];
    let right_char_end = chars.len();

    let mut out: Vec<(TextBlock, Vec<CharBBox>)> = Vec::with_capacity(3);
    push_split_segment(
        &mut out,
        block,
        &block.text[..match_start],
        chars,
        0,
        left_char_end,
    );
    push_split_segment(
        &mut out,
        block,
        &block.text[match_start..match_end],
        chars,
        match_char_start,
        match_char_end,
    );
    push_split_segment(
        &mut out,
        block,
        &block.text[match_end..],
        chars,
        right_char_start,
        right_char_end,
    );

    if out.is_empty() {
        return None;
    }
    Some(out)
}

fn push_split_segment(
    out: &mut Vec<(TextBlock, Vec<CharBBox>)>,
    source: &TextBlock,
    text: &str,
    chars: &[CharBBox],
    char_start: usize,
    char_end: usize,
) {
    if text.is_empty() || char_start >= char_end || char_end > chars.len() {
        return;
    }
    let segment_chars = chars[char_start..char_end].to_vec();
    let Some(segment_bbox) = bbox_from_chars(&segment_chars) else {
        return;
    };
    out.push((
        TextBlock {
            page_index: source.page_index,
            text: text.to_string(),
            bbox: segment_bbox,
        },
        segment_chars,
    ));
}

fn bbox_from_chars(chars: &[CharBBox]) -> Option<Rectangle> {
    let mut iter = chars.iter();
    let first = iter.next()?;
    let mut bbox = first.bbox;
    for ch in iter {
        bbox = union_rectangles(bbox, ch.bbox);
    }
    Some(bbox)
}

fn map_text_char_boundaries_to_char_offsets(text: &str, chars: &[CharBBox]) -> Option<Vec<usize>> {
    let mut boundaries: Vec<usize> = Vec::with_capacity(text.chars().count().saturating_add(1));
    boundaries.push(0);
    let mut consumed = 0usize;

    for ch in text.chars() {
        if ch.is_whitespace() {
            boundaries.push(consumed);
            continue;
        }
        while consumed < chars.len() && chars[consumed].ch != ch {
            consumed += 1;
        }
        if consumed >= chars.len() {
            return None;
        }
        consumed += 1;
        boundaries.push(consumed);
    }

    Some(boundaries)
}

fn byte_range_to_char_range(text: &str, start: usize, end: usize) -> Option<(usize, usize)> {
    if start > end || end > text.len() {
        return None;
    }
    if !text.is_char_boundary(start) || !text.is_char_boundary(end) {
        return None;
    }

    let mut start_char = None;
    let mut end_char = None;
    let mut count = 0usize;

    if start == 0 {
        start_char = Some(0);
    }
    if end == 0 {
        end_char = Some(0);
    }

    for (idx, _) in text.char_indices() {
        if idx == start {
            start_char = Some(count);
        }
        if idx == end {
            end_char = Some(count);
        }
        count += 1;
    }

    if start == text.len() {
        start_char = Some(count);
    }
    if end == text.len() {
        end_char = Some(count);
    }

    Some((start_char?, end_char?))
}

fn union_rectangles(a: Rectangle, b: Rectangle) -> Rectangle {
    Rectangle {
        top: a.top.min(b.top),
        left: a.left.min(b.left),
        bottom: a.bottom.max(b.bottom),
        right: a.right.max(b.right),
    }
}

#[cfg(test)]
mod tests {
    use super::{
        MAX_DENSE_INFERRED_PAGE_RECTS, ScopedMergeRules, SplitError, TextBlockIndex, is_date_token,
        is_numeric_token,
    };
    use crate::text::{CharBBox, Rectangle, TextBlock};

    #[test]
    fn numeric_token_detector_matches_expected_cases() {
        for value in [
            "-1624.57",
            "234.52",
            "$-100",
            "-$100",
            "(1,234.00)",
            "123-",
            ".50",
            "1,234,567.89",
        ] {
            assert!(is_numeric_token(value), "expected numeric token: {}", value);
        }
    }

    #[test]
    fn numeric_token_detector_rejects_non_numeric_cases() {
        for value in ["", "abc", "12,34", "1,23,456", "$", ".", "-()", "1-2-3"] {
            assert!(
                !is_numeric_token(value),
                "expected non-numeric token: {}",
                value
            );
        }
    }

    #[test]
    fn date_token_detector_matches_expected_cases() {
        for value in [
            "1/2/24",
            "01-02-2024",
            "(1/2/2024)",
            "1/2/24-12/31/2024",
            "01-02-24-12-31-2024",
        ] {
            assert!(is_date_token(value), "expected date token: {}", value);
        }
    }

    #[test]
    fn date_token_detector_rejects_non_date_cases() {
        for value in ["", "2024/01/02", "1/2", "1/2/20245", "1//2024", "abc"] {
            assert!(!is_date_token(value), "expected non-date token: {}", value);
        }
    }

    fn make_block(
        page_index: usize,
        text: &str,
        top: f64,
        left: f64,
        bottom: f64,
        right: f64,
    ) -> TextBlock {
        TextBlock {
            page_index,
            text: text.to_string(),
            bbox: Rectangle {
                top,
                left,
                bottom,
                right,
            },
        }
    }

    fn make_char(
        page_index: usize,
        ch: char,
        top: f64,
        left: f64,
        bottom: f64,
        right: f64,
    ) -> CharBBox {
        CharBBox {
            page_index,
            ch,
            bbox: Rectangle {
                top,
                left,
                bottom,
                right,
            },
        }
    }

    #[test]
    fn scoped_merge_concatenates_chars_without_synthetic_spaces() {
        let blocks = vec![
            make_block(0, "AB", 10.0, 10.0, 20.0, 20.0),
            make_block(0, "CD", 10.0, 21.0, 20.0, 30.0),
        ];
        let block_chars = vec![
            vec![
                make_char(0, 'A', 10.0, 10.0, 20.0, 14.0),
                make_char(0, 'B', 10.0, 14.0, 20.0, 20.0),
            ],
            vec![
                make_char(0, 'C', 10.0, 21.0, 20.0, 25.0),
                make_char(0, 'D', 10.0, 25.0, 20.0, 30.0),
            ],
        ];
        let index = TextBlockIndex::new_with_chars(blocks, block_chars);
        let scoped = index.scoped(
            Rectangle {
                top: 0.0,
                left: 0.0,
                bottom: 100.0,
                right: 100.0,
            },
            0.0,
            5.0,
            false,
            ScopedMergeRules::default(),
        );
        assert_eq!(scoped.block_len(), 1);
        assert_eq!(
            scoped.block_at(0).map(|block| block.text.as_str()),
            Some("AB CD")
        );
        let chars = scoped
            .block_chars_at(0)
            .expect("expected merged block chars to be present")
            .iter()
            .map(|ch| ch.ch)
            .collect::<String>();
        assert_eq!(chars, "ABCD");
    }

    #[test]
    fn scoped_normalize_preserves_char_attachment_per_block() {
        let blocks = vec![
            make_block(0, "LOW", 40.0, 10.0, 50.0, 20.0),
            make_block(0, "HIGH", 10.0, 10.0, 20.0, 20.0),
        ];
        let block_chars = vec![
            vec![
                make_char(0, 'L', 40.0, 10.0, 50.0, 13.0),
                make_char(0, 'W', 40.0, 17.0, 50.0, 20.0),
            ],
            vec![
                make_char(0, 'H', 10.0, 10.0, 20.0, 13.0),
                make_char(0, 'I', 10.0, 13.0, 20.0, 16.0),
            ],
        ];
        let index = TextBlockIndex::new_with_chars(blocks, block_chars);
        let scoped = index.scoped(
            Rectangle {
                top: 0.0,
                left: 0.0,
                bottom: 100.0,
                right: 100.0,
            },
            0.0,
            0.0,
            true,
            ScopedMergeRules::default(),
        );
        assert_eq!(
            scoped.block_at(0).map(|block| block.text.as_str()),
            Some("HIGH")
        );
        assert_eq!(
            scoped.block_at(1).map(|block| block.text.as_str()),
            Some("LOW")
        );
        let first_chars = scoped
            .block_chars_at(0)
            .expect("expected chars for first normalized block")
            .iter()
            .map(|ch| ch.ch)
            .collect::<String>();
        let second_chars = scoped
            .block_chars_at(1)
            .expect("expected chars for second normalized block")
            .iter()
            .map(|ch| ch.ch)
            .collect::<String>();
        assert_eq!(first_chars, "HI");
        assert_eq!(second_chars, "LW");
    }

    #[test]
    fn split_splits_single_block_into_left_match_and_right() {
        let blocks = vec![make_block(0, "He123llo", 10.0, 10.0, 20.0, 38.0)];
        let block_chars = vec![vec![
            make_char(0, 'H', 10.0, 10.0, 20.0, 14.0),
            make_char(0, 'e', 10.0, 14.0, 20.0, 18.0),
            make_char(0, '1', 10.0, 18.0, 20.0, 22.0),
            make_char(0, '2', 10.0, 22.0, 20.0, 26.0),
            make_char(0, '3', 10.0, 26.0, 20.0, 30.0),
            make_char(0, 'l', 10.0, 30.0, 20.0, 34.0),
            make_char(0, 'l', 10.0, 34.0, 20.0, 36.0),
            make_char(0, 'o', 10.0, 36.0, 20.0, 38.0),
        ]];
        let index = TextBlockIndex::new_with_chars(blocks, block_chars);
        let split = index.split("123").expect("split should succeed");
        assert_eq!(split.block_len(), 3);
        assert_eq!(
            split.block_at(0).map(|block| block.text.as_str()),
            Some("He")
        );
        assert_eq!(
            split.block_at(1).map(|block| block.text.as_str()),
            Some("123")
        );
        assert_eq!(
            split.block_at(2).map(|block| block.text.as_str()),
            Some("llo")
        );
        assert_eq!(split.block_at(0).map(|block| block.bbox.left), Some(10.0));
        assert_eq!(split.block_at(0).map(|block| block.bbox.right), Some(18.0));
        assert_eq!(split.block_at(1).map(|block| block.bbox.left), Some(18.0));
        assert_eq!(split.block_at(1).map(|block| block.bbox.right), Some(30.0));
        assert_eq!(split.block_at(2).map(|block| block.bbox.left), Some(30.0));
        assert_eq!(split.block_at(2).map(|block| block.bbox.right), Some(38.0));
        assert_eq!(
            split
                .block_chars_at(0)
                .expect("left chars should exist")
                .iter()
                .map(|ch| ch.ch)
                .collect::<String>(),
            "He"
        );
        assert_eq!(
            split
                .block_chars_at(1)
                .expect("match chars should exist")
                .iter()
                .map(|ch| ch.ch)
                .collect::<String>(),
            "123"
        );
        assert_eq!(
            split
                .block_chars_at(2)
                .expect("right chars should exist")
                .iter()
                .map(|ch| ch.ch)
                .collect::<String>(),
            "llo"
        );
    }

    #[test]
    fn split_uses_first_non_empty_match_per_block() {
        let blocks = vec![make_block(0, "a1b1c", 10.0, 10.0, 20.0, 35.0)];
        let block_chars = vec![vec![
            make_char(0, 'a', 10.0, 10.0, 20.0, 15.0),
            make_char(0, '1', 10.0, 15.0, 20.0, 20.0),
            make_char(0, 'b', 10.0, 20.0, 20.0, 25.0),
            make_char(0, '1', 10.0, 25.0, 20.0, 30.0),
            make_char(0, 'c', 10.0, 30.0, 20.0, 35.0),
        ]];
        let index = TextBlockIndex::new_with_chars(blocks, block_chars);
        let split = index.split("1").expect("split should succeed");
        assert_eq!(split.block_len(), 3);
        assert_eq!(
            split.block_at(0).map(|block| block.text.as_str()),
            Some("a")
        );
        assert_eq!(
            split.block_at(1).map(|block| block.text.as_str()),
            Some("1")
        );
        assert_eq!(
            split.block_at(2).map(|block| block.text.as_str()),
            Some("b1c")
        );
    }

    #[test]
    fn split_preserves_non_matching_blocks_and_order() {
        let blocks = vec![
            make_block(0, "alpha", 10.0, 10.0, 20.0, 20.0),
            make_block(0, "He123llo", 10.0, 30.0, 20.0, 58.0),
            make_block(1, "omega", 40.0, 10.0, 50.0, 25.0),
        ];
        let block_chars = vec![
            vec![
                make_char(0, 'a', 10.0, 10.0, 20.0, 12.0),
                make_char(0, 'l', 10.0, 12.0, 20.0, 14.0),
                make_char(0, 'p', 10.0, 14.0, 20.0, 16.0),
                make_char(0, 'h', 10.0, 16.0, 20.0, 18.0),
                make_char(0, 'a', 10.0, 18.0, 20.0, 20.0),
            ],
            vec![
                make_char(0, 'H', 10.0, 30.0, 20.0, 34.0),
                make_char(0, 'e', 10.0, 34.0, 20.0, 38.0),
                make_char(0, '1', 10.0, 38.0, 20.0, 42.0),
                make_char(0, '2', 10.0, 42.0, 20.0, 46.0),
                make_char(0, '3', 10.0, 46.0, 20.0, 50.0),
                make_char(0, 'l', 10.0, 50.0, 20.0, 53.0),
                make_char(0, 'l', 10.0, 53.0, 20.0, 55.0),
                make_char(0, 'o', 10.0, 55.0, 20.0, 58.0),
            ],
            vec![
                make_char(1, 'o', 40.0, 10.0, 50.0, 13.0),
                make_char(1, 'm', 40.0, 13.0, 50.0, 16.0),
                make_char(1, 'e', 40.0, 16.0, 50.0, 19.0),
                make_char(1, 'g', 40.0, 19.0, 50.0, 22.0),
                make_char(1, 'a', 40.0, 22.0, 50.0, 25.0),
            ],
        ];
        let index = TextBlockIndex::new_with_chars(blocks, block_chars);
        let split = index.split("123").expect("split should succeed");
        let texts = (0..split.block_len())
            .filter_map(|i| split.block_at(i).map(|block| block.text.clone()))
            .collect::<Vec<String>>();
        assert_eq!(texts, vec!["alpha", "He", "123", "llo", "omega"]);
        assert_eq!(split.block_at(0).map(|block| block.page_index), Some(0));
        assert_eq!(split.block_at(4).map(|block| block.page_index), Some(1));
    }

    #[test]
    fn split_handles_synthetic_spaces_from_scoped_merges() {
        let blocks = vec![
            make_block(0, "AB", 10.0, 10.0, 20.0, 20.0),
            make_block(0, "CD", 10.0, 21.0, 20.0, 30.0),
        ];
        let block_chars = vec![
            vec![
                make_char(0, 'A', 10.0, 10.0, 20.0, 14.0),
                make_char(0, 'B', 10.0, 14.0, 20.0, 20.0),
            ],
            vec![
                make_char(0, 'C', 10.0, 21.0, 20.0, 25.0),
                make_char(0, 'D', 10.0, 25.0, 20.0, 30.0),
            ],
        ];
        let index = TextBlockIndex::new_with_chars(blocks, block_chars);
        let scoped = index.scoped(
            Rectangle {
                top: 0.0,
                left: 0.0,
                bottom: 100.0,
                right: 100.0,
            },
            0.0,
            5.0,
            false,
            ScopedMergeRules::default(),
        );
        assert_eq!(
            scoped.block_at(0).map(|block| block.text.as_str()),
            Some("AB CD")
        );

        let split = scoped.split("CD").expect("split should succeed");
        assert_eq!(split.block_len(), 2);
        assert_eq!(
            split.block_at(0).map(|block| block.text.as_str()),
            Some("AB ")
        );
        assert_eq!(
            split.block_at(1).map(|block| block.text.as_str()),
            Some("CD")
        );
        assert_eq!(
            split
                .block_chars_at(0)
                .expect("left chars should exist")
                .iter()
                .map(|ch| ch.ch)
                .collect::<String>(),
            "AB"
        );
        assert_eq!(
            split
                .block_chars_at(1)
                .expect("match chars should exist")
                .iter()
                .map(|ch| ch.ch)
                .collect::<String>(),
            "CD"
        );
    }

    #[test]
    fn split_returns_error_when_block_chars_are_missing() {
        let index = TextBlockIndex::new(vec![make_block(0, "He123llo", 10.0, 10.0, 20.0, 38.0)]);
        let err = index
            .split("123")
            .expect_err("split should fail without block chars");
        assert!(matches!(err, SplitError::MissingBlockChars));
    }

    #[test]
    fn split_invalid_pattern_returns_error() {
        let blocks = vec![make_block(0, "He123llo", 10.0, 10.0, 20.0, 38.0)];
        let block_chars = vec![vec![
            make_char(0, 'H', 10.0, 10.0, 20.0, 14.0),
            make_char(0, 'e', 10.0, 14.0, 20.0, 18.0),
            make_char(0, '1', 10.0, 18.0, 20.0, 22.0),
            make_char(0, '2', 10.0, 22.0, 20.0, 26.0),
            make_char(0, '3', 10.0, 26.0, 20.0, 30.0),
            make_char(0, 'l', 10.0, 30.0, 20.0, 34.0),
            make_char(0, 'l', 10.0, 34.0, 20.0, 36.0),
            make_char(0, 'o', 10.0, 36.0, 20.0, 38.0),
        ]];
        let index = TextBlockIndex::new_with_chars(blocks, block_chars);
        let err = index
            .split("(")
            .expect_err("pattern should be reported as invalid");
        assert!(matches!(err, SplitError::InvalidPattern(_)));
    }

    #[test]
    fn page_rects_are_inferred_per_page_index_when_not_provided() {
        let blocks = vec![
            make_block(0, "P0-A", 10.0, 10.0, 20.0, 25.0),
            make_block(0, "P0-B", 18.0, 30.0, 28.0, 40.0),
            make_block(2, "P2-A", 220.0, 5.0, 230.0, 20.0),
        ];
        let index = TextBlockIndex::new(blocks);
        assert_eq!(
            index.page_rects(),
            vec![
                Rectangle {
                    top: 10.0,
                    left: 10.0,
                    bottom: 28.0,
                    right: 40.0,
                },
                Rectangle {
                    top: 0.0,
                    left: 0.0,
                    bottom: 0.0,
                    right: 0.0,
                },
                Rectangle {
                    top: 220.0,
                    left: 5.0,
                    bottom: 230.0,
                    right: 20.0,
                },
            ]
        );
        assert_eq!(
            index.doc_rect(),
            Rectangle {
                top: 10.0,
                left: 5.0,
                bottom: 230.0,
                right: 40.0,
            }
        );
    }

    #[test]
    fn page_rects_preserve_explicit_values() {
        let blocks = vec![
            make_block(0, "P0", 10.0, 10.0, 20.0, 20.0),
            make_block(1, "P1", 110.0, 10.0, 120.0, 20.0),
        ];
        let block_chars = vec![
            vec![make_char(0, 'A', 10.0, 10.0, 20.0, 20.0)],
            vec![make_char(1, 'B', 110.0, 10.0, 120.0, 20.0)],
        ];
        let page_rects = vec![
            Rectangle {
                top: 0.0,
                left: 0.0,
                bottom: 100.0,
                right: 80.0,
            },
            Rectangle {
                top: 100.0,
                left: 0.0,
                bottom: 210.0,
                right: 80.0,
            },
        ];
        let index = TextBlockIndex::new_with_chars_and_page_rects(blocks, block_chars, page_rects);
        assert_eq!(
            index.doc_rect(),
            Rectangle {
                top: 0.0,
                left: 0.0,
                bottom: 210.0,
                right: 80.0,
            }
        );
        assert_eq!(index.page_rects()[1].top, 100.0);
        assert_eq!(index.page_rects()[1].bottom, 210.0);
    }

    #[test]
    fn page_rect_inference_uses_compact_fallback_for_huge_sparse_indices() {
        let blocks = vec![
            make_block(0, "P0", 10.0, 10.0, 20.0, 20.0),
            make_block(
                MAX_DENSE_INFERRED_PAGE_RECTS + 10,
                "PX",
                210.0,
                10.0,
                220.0,
                20.0,
            ),
        ];
        let index = TextBlockIndex::new(blocks);
        assert_eq!(index.page_rects().len(), 2);
        assert_eq!(
            index.doc_rect(),
            Rectangle {
                top: 10.0,
                left: 10.0,
                bottom: 220.0,
                right: 20.0,
            }
        );
    }
}
