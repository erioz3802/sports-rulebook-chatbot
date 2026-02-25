/**
 * Rule-aware chunking for baseball rulebooks.
 * Splits text at rule/section boundaries instead of fixed character counts.
 */

// Patterns that indicate rule/section boundaries
const BOUNDARY_PATTERNS = [
  // NFHS style: "1-1-1", "8-4-2", etc. at start of line or after whitespace
  /(?:^|\n)\s*(\d{1,2}-\d{1,2}-\d{1,2}[a-z]?)\b/,
  // NCAA style: "Rule 8-3b", "RULE 5", etc.
  /(?:^|\n)\s*(RULE\s+\d+[-.]?\d*[a-z]?)/i,
  // MLB/OBR style: "5.06(b)", "6.01(a)(1)", etc.
  /(?:^|\n)\s*(\d+\.\d{2}\s*\([a-z]\))/,
  // Section headers: "SECTION 1", "Section 4", "ART. 1"
  /(?:^|\n)\s*((?:SECTION|ART\.?)\s+\d+)/i,
  // Penalty/Exception/Note markers
  /(?:^|\n)\s*(PENALTY|EXCEPTION|NOTE|APPROVED RULING|A\.R\.)\s*[:.\-—]/i,
  // Page markers from PDF extraction
  /\[Page\s+\d+\]/,
];

// Combined regex for splitting
const BOUNDARY_REGEX = new RegExp(
  BOUNDARY_PATTERNS.map(p => p.source).join('|'),
  'gim'
);

// Detect the rule reference at the start of a text block
const RULE_REF_PATTERNS = [
  /^[\s\n]*(\d{1,2}-\d{1,2}-\d{1,2}[a-z]?)\b/,
  /^[\s\n]*(RULE\s+\d+[-.]?\d*[a-z]?)/i,
  /^[\s\n]*(\d+\.\d{2}\s*\([a-z]\)(?:\(\d+\))?)/,
  /^[\s\n]*((?:SECTION|ART\.?)\s+\d+)/i,
];

// Detect section type
const SECTION_TYPE_PATTERNS = [
  { type: 'penalty', pattern: /\bPENALTY\b/i },
  { type: 'exception', pattern: /\bEXCEPTION\b/i },
  { type: 'note', pattern: /\b(?:NOTE|A\.R\.|APPROVED RULING)\b/i },
  { type: 'definition', pattern: /\b(?:DEFINITION|DEFINITIONS)\b/i },
];

function extractRuleRef(text) {
  for (const pattern of RULE_REF_PATTERNS) {
    const match = text.match(pattern);
    if (match) return match[1].trim();
  }
  return null;
}

function detectSectionType(text) {
  for (const { type, pattern } of SECTION_TYPE_PATTERNS) {
    if (pattern.test(text)) return type;
  }
  return 'rule';
}

function extractPageNumber(text) {
  const match = text.match(/\[Page\s+(\d+)\]/);
  return match ? parseInt(match[1], 10) : null;
}

// Track the last seen page number across chunks
function getPageFromText(text) {
  const pages = [...text.matchAll(/\[Page\s+(\d+)\]/g)];
  return pages.length > 0 ? parseInt(pages[pages.length - 1][1], 10) : null;
}

/**
 * Split text at sentence boundaries, respecting a max size.
 */
function splitAtSentences(text, maxSize) {
  if (text.length <= maxSize) return [text];

  const chunks = [];
  const sentences = text.split(/(?<=[.!?])\s+/);
  let current = '';

  for (const sentence of sentences) {
    if (current.length + sentence.length + 1 > maxSize && current.length > 0) {
      chunks.push(current.trim());
      current = '';
    }
    current += (current ? ' ' : '') + sentence;
  }

  if (current.trim()) {
    chunks.push(current.trim());
  }

  // If we still have oversized chunks, hard-split them
  const result = [];
  for (const chunk of chunks) {
    if (chunk.length > maxSize * 1.5) {
      for (let i = 0; i < chunk.length; i += maxSize) {
        result.push(chunk.slice(i, i + maxSize));
      }
    } else {
      result.push(chunk);
    }
  }

  return result;
}

/**
 * Rule-aware text chunking.
 * Splits at rule boundaries, enriches chunks with metadata.
 *
 * @param {string} fullText - The complete document text
 * @param {object} options
 * @param {number} options.maxChunkSize - Maximum characters per chunk (default 2000)
 * @param {number} options.minChunkSize - Minimum characters per chunk, merge smaller ones (default 200)
 * @returns {Array<{text: string, ruleRef: string|null, sectionType: string, pageNumber: number|null}>}
 */
export const chunkText = (fullText, { maxChunkSize = 2000, minChunkSize = 200 } = {}) => {
  if (!fullText || fullText.trim().length === 0) return [];

  // Find all boundary positions
  const boundaries = [];
  let match;
  const regex = new RegExp(BOUNDARY_REGEX.source, 'gim');

  while ((match = regex.exec(fullText)) !== null) {
    boundaries.push(match.index);
  }

  // Add start and end
  if (boundaries.length === 0 || boundaries[0] !== 0) {
    boundaries.unshift(0);
  }

  // Deduplicate and sort
  const uniqueBoundaries = [...new Set(boundaries)].sort((a, b) => a - b);

  // Split at boundaries
  const rawSections = [];
  for (let i = 0; i < uniqueBoundaries.length; i++) {
    const start = uniqueBoundaries[i];
    const end = i + 1 < uniqueBoundaries.length ? uniqueBoundaries[i + 1] : fullText.length;
    const text = fullText.slice(start, end).trim();
    if (text.length > 0) {
      rawSections.push(text);
    }
  }

  // Process sections: split oversized ones, merge tiny ones
  let lastPage = null;
  const enrichedChunks = [];

  for (const section of rawSections) {
    const pageInSection = getPageFromText(section);
    if (pageInSection !== null) lastPage = pageInSection;

    const pieces = section.length > maxChunkSize
      ? splitAtSentences(section, maxChunkSize)
      : [section];

    for (const piece of pieces) {
      const piecePageNumber = extractPageNumber(piece) || lastPage;
      enrichedChunks.push({
        text: piece,
        ruleRef: extractRuleRef(piece) || extractRuleRef(section),
        sectionType: detectSectionType(piece),
        pageNumber: piecePageNumber,
      });
    }
  }

  // Merge chunks that are too small with the next chunk
  const merged = [];
  let buffer = null;

  for (const chunk of enrichedChunks) {
    if (buffer) {
      if (buffer.text.length < minChunkSize) {
        // Merge with current chunk
        buffer.text = buffer.text + '\n\n' + chunk.text;
        buffer.ruleRef = buffer.ruleRef || chunk.ruleRef;
        buffer.sectionType = chunk.sectionType !== 'rule' ? chunk.sectionType : buffer.sectionType;
        buffer.pageNumber = chunk.pageNumber || buffer.pageNumber;
        continue;
      } else {
        merged.push(buffer);
      }
    }
    buffer = { ...chunk };
  }
  if (buffer) merged.push(buffer);

  return merged;
};
