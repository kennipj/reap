use std::collections::HashMap;
use std::collections::VecDeque;
use std::sync::{Mutex, OnceLock};

use regex::RegexBuilder;

use crate::text::{Rectangle, TextBlock};

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
    rects: Vec<RectF>,
    rtree: RTree,
    cache: LruCache,
    regex_cache: RegexLruCache,
    regex_indices_cache: RegexIndicesLruCache,
    regex_index: RegexSearchIndex,
    doc_rect: Rectangle,
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

impl TextBlockIndex {
    pub fn new(blocks: Vec<TextBlock>) -> Self {
        Self::from_parts(blocks)
    }

    fn from_parts(blocks: Vec<TextBlock>) -> Self {
        let regex_index = build_regex_index(&blocks);
        let rects: Vec<RectF> = blocks
            .iter()
            .map(|b| RectF {
                min_x: b.bbox.left,
                min_y: b.bbox.top,
                max_x: b.bbox.right,
                max_y: b.bbox.bottom,
            })
            .collect();
        let doc_rect = doc_rect_from_rects(&rects);
        let rtree = RTree::build(&rects, 16);
        Self {
            blocks,
            rects,
            rtree,
            cache: LruCache::new(1024),
            regex_cache: RegexLruCache::new(128),
            regex_indices_cache: RegexIndicesLruCache::new(128),
            regex_index,
            doc_rect,
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
        let mut blocks: Vec<TextBlock> = indices
            .into_iter()
            .map(|index| self.blocks[index].clone())
            .collect();
        if normalize {
            blocks = normalize_scoped_blocks(blocks);
        }
        if merge_threshold > 0.0 {
            blocks = merge_scoped_blocks(blocks, merge_threshold, rules);
        }
        Self::from_parts(blocks)
    }

    pub fn blocks(&self) -> Vec<TextBlock> {
        self.blocks.clone()
    }

    pub fn block_at(&self, index: usize) -> Option<&TextBlock> {
        self.blocks.get(index)
    }

    pub fn block_len(&self) -> usize {
        self.blocks.len()
    }

    pub fn text(&self) -> String {
        let mut out = String::new();
        for page in &self.regex_index.pages {
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

    pub fn search_regex_indices(
        &mut self,
        pattern: &str,
    ) -> Result<Vec<Vec<usize>>, RegexSearchError> {
        if let Some(cached) = self.regex_indices_cache.get(pattern) {
            return Ok(cached);
        }

        let regex = compiled_regex(pattern).map_err(RegexSearchError::InvalidPattern)?;
        let mut out: Vec<Vec<usize>> = Vec::new();
        for page in &self.regex_index.pages {
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

    pub fn doc_rect(&self) -> Rectangle {
        self.doc_rect
    }

    fn matching_block_indices(&self, rect: Rectangle, overlap: f64) -> Vec<usize> {
        let overlap = overlap.clamp(0.0, 1.0);
        let query = RectF::from_ltrb(rect.left, rect.top, rect.right, rect.bottom);
        self.rtree.search(&self.rects, &query, overlap)
    }
}

fn doc_rect_from_rects(rects: &[RectF]) -> Rectangle {
    if rects.is_empty() {
        return Rectangle {
            top: 0.0,
            left: 0.0,
            bottom: 0.0,
            right: 0.0,
        };
    }
    let mut min_x = rects[0].min_x;
    let mut min_y = rects[0].min_y;
    let mut max_x = rects[0].max_x;
    let mut max_y = rects[0].max_y;
    for r in rects.iter().skip(1) {
        min_x = min_x.min(r.min_x);
        min_y = min_y.min(r.min_y);
        max_x = max_x.max(r.max_x);
        max_y = max_y.max(r.max_y);
    }
    Rectangle {
        top: min_y,
        left: min_x,
        bottom: max_y,
        right: max_x,
    }
}

fn normalize_scoped_blocks(blocks: Vec<TextBlock>) -> Vec<TextBlock> {
    if blocks.len() < 2 {
        return blocks;
    }

    let mut per_page: HashMap<usize, Vec<TextBlock>> = HashMap::new();
    for block in blocks {
        per_page.entry(block.page_index).or_default().push(block);
    }

    let mut page_ids: Vec<usize> = per_page.keys().copied().collect();
    page_ids.sort_unstable();

    let mut normalized: Vec<TextBlock> = Vec::new();
    for page_id in page_ids {
        let page_blocks = per_page.remove(&page_id).unwrap_or_default();
        normalized.extend(normalize_page_blocks(page_blocks));
    }

    sort_blocks_for_reading_order(&mut normalized);
    normalized
}

fn normalize_page_blocks(mut blocks: Vec<TextBlock>) -> Vec<TextBlock> {
    if blocks.len() < 2 {
        return blocks;
    }

    blocks.sort_by(|a, b| {
        a.bbox
            .top
            .total_cmp(&b.bbox.top)
            .then_with(|| a.bbox.left.total_cmp(&b.bbox.left))
            .then_with(|| a.bbox.right.total_cmp(&b.bbox.right))
    });

    let mut normalized: Vec<TextBlock> = Vec::with_capacity(blocks.len());
    let mut current_line: Vec<TextBlock> = Vec::new();

    for block in blocks {
        if current_line.is_empty() {
            current_line.push(block);
            continue;
        }

        let anchor = &current_line[0];
        let same_line = anchor.bbox.vertical_overlap(&block.bbox, 0.5)
            || block.bbox.vertical_overlap(&anchor.bbox, 0.5);
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

fn normalize_line_blocks(mut line: Vec<TextBlock>) -> Vec<TextBlock> {
    if line.len() < 2 {
        return line;
    }

    let mut top_sum = 0.0;
    let mut bottom_sum = 0.0;
    for block in &line {
        top_sum += block.bbox.top.min(block.bbox.bottom);
        bottom_sum += block.bbox.top.max(block.bbox.bottom);
    }
    let n = line.len() as f64;
    let avg_top = top_sum / n;
    let avg_bottom = bottom_sum / n;

    for block in &mut line {
        block.bbox.top = avg_top;
        block.bbox.bottom = avg_bottom;
    }

    line
}

fn merge_scoped_blocks(
    blocks: Vec<TextBlock>,
    merge_threshold: f64,
    rules: ScopedMergeRules,
) -> Vec<TextBlock> {
    if blocks.len() < 2 {
        return blocks;
    }

    let merge_threshold = merge_threshold.max(0.0);
    if merge_threshold <= 0.0 {
        return blocks;
    }

    let mut per_page: HashMap<usize, Vec<TextBlock>> = HashMap::new();
    for block in blocks {
        per_page.entry(block.page_index).or_default().push(block);
    }

    let mut page_ids: Vec<usize> = per_page.keys().copied().collect();
    page_ids.sort_unstable();

    let mut merged: Vec<TextBlock> = Vec::new();
    for page_id in page_ids {
        let page_blocks = per_page.remove(&page_id).unwrap_or_default();
        merged.extend(merge_page_blocks(page_blocks, merge_threshold, rules));
    }

    sort_blocks_for_reading_order(&mut merged);
    merged
}

fn sort_blocks_for_reading_order(blocks: &mut [TextBlock]) {
    blocks.sort_by(|a, b| {
        a.page_index
            .cmp(&b.page_index)
            .then_with(|| a.bbox.top.total_cmp(&b.bbox.top))
            .then_with(|| a.bbox.left.total_cmp(&b.bbox.left))
            .then_with(|| a.bbox.right.total_cmp(&b.bbox.right))
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
    blocks: Vec<TextBlock>,
    merge_threshold: f64,
    rules: ScopedMergeRules,
) -> Vec<TextBlock> {
    if blocks.len() < 2 {
        return blocks;
    }

    let classes: Vec<BlockClass> = blocks
        .iter()
        .map(|block| classify_block_text(&block.text))
        .collect();
    let spans: Vec<(f64, f64)> = blocks
        .iter()
        .map(|block| {
            (
                block.bbox.left.min(block.bbox.right),
                block.bbox.left.max(block.bbox.right),
            )
        })
        .collect();
    let mut order: Vec<usize> = (0..blocks.len()).collect();
    order.sort_by(|&a, &b| {
        spans[a]
            .0
            .total_cmp(&spans[b].0)
            .then_with(|| spans[a].1.total_cmp(&spans[b].1))
            .then_with(|| blocks[a].bbox.top.total_cmp(&blocks[b].bbox.top))
            .then_with(|| a.cmp(&b))
    });

    let mut dsu = DisjointSet::new(blocks.len());
    let mut active: Vec<usize> = Vec::new();
    for curr in order {
        let curr_left = spans[curr].0;
        active.retain(|&candidate| spans[candidate].1 + merge_threshold >= curr_left);
        for &candidate in &active {
            if blocks_are_mergeable(
                &blocks[curr],
                classes[curr],
                &blocks[candidate],
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

    let mut merged: Vec<TextBlock> = Vec::with_capacity(components.len());
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

fn merge_component(blocks: &[TextBlock], component: &[usize]) -> TextBlock {
    let mut members: Vec<&TextBlock> = component.iter().map(|&index| &blocks[index]).collect();
    members.sort_by(|a, b| {
        a.bbox
            .left
            .total_cmp(&b.bbox.left)
            .then_with(|| a.bbox.top.total_cmp(&b.bbox.top))
            .then_with(|| a.bbox.right.total_cmp(&b.bbox.right))
    });

    let first = members[0];
    let mut text = String::new();
    let mut bbox = first.bbox;
    for (idx, block) in members.iter().enumerate() {
        if idx > 0 {
            text.push(' ');
        }
        text.push_str(&block.text);
        bbox = union_rectangles(bbox, block.bbox);
    }
    TextBlock {
        page_index: first.page_index,
        text,
        bbox,
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
        let mut block_indices = per_page.remove(&page_id).unwrap_or_default();
        block_indices.sort_by(|&a, &b| {
            let ba = &blocks[a];
            let bb = &blocks[b];
            ba.bbox
                .top
                .total_cmp(&bb.bbox.top)
                .then_with(|| ba.bbox.left.total_cmp(&bb.bbox.left))
                .then_with(|| ba.bbox.right.total_cmp(&bb.bbox.right))
        });
        let lines = cluster_block_indices_into_lines(blocks, &block_indices);
        let (body, spans) = build_page_body(blocks, &lines);
        pages.push(RegexPageIndex { body, spans });
    }

    RegexSearchIndex { pages }
}

fn cluster_block_indices_into_lines(
    blocks: &[TextBlock],
    sorted_block_indices: &[usize],
) -> Vec<Vec<usize>> {
    let mut lines: Vec<Vec<usize>> = Vec::new();
    let mut current_line: Vec<usize> = Vec::new();

    for &block_index in sorted_block_indices {
        let block = &blocks[block_index];
        if current_line.is_empty() {
            current_line.push(block_index);
            continue;
        }

        if blocks[current_line[0]]
            .bbox
            .vertical_overlap(&block.bbox, 0.5)
        {
            current_line.push(block_index);
            continue;
        }

        current_line.sort_by(|&a, &b| {
            let ba = &blocks[a];
            let bb = &blocks[b];
            ba.bbox
                .left
                .total_cmp(&bb.bbox.left)
                .then_with(|| ba.bbox.top.total_cmp(&bb.bbox.top))
        });
        lines.push(current_line);
        current_line = vec![block_index];
    }

    if !current_line.is_empty() {
        current_line.sort_by(|&a, &b| {
            let ba = &blocks[a];
            let bb = &blocks[b];
            ba.bbox
                .left
                .total_cmp(&bb.bbox.left)
                .then_with(|| ba.bbox.top.total_cmp(&bb.bbox.top))
        });
        lines.push(current_line);
    }

    lines
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
    use super::{is_date_token, is_numeric_token};

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
}
