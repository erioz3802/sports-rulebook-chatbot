import { extractKeywords } from './keywords.js';
import { BM25Index } from './bm25.js';
import { expandWithSynonyms } from './synonyms.js';
import { enhanceQuery } from './enhancer.js';

export { BM25Index } from './bm25.js';
export { extractKeywords } from './keywords.js';

// Rule reference patterns to detect in queries
const RULE_PATTERNS = [
  /\b(\d{1,2}-\d{1,2}-\d{1,2}[a-z]?)\b/gi,
  /\b(rule\s+\d+[-.]?\d*[a-z]?)\b/gi,
  /\b(\d+\.\d{2}\s*\([a-z]\)(?:\(\d+\))?)\b/gi,
];

// Intent patterns that map to section types
const INTENT_PATTERNS = [
  { type: 'penalty', patterns: [/\bpenalt(?:y|ies)\b/i, /\bpunish/i, /\bconsequence/i] },
  { type: 'exception', patterns: [/\bexception/i, /\bexcept\b/i, /\bunless\b/i] },
  { type: 'note', patterns: [/\bnote\b/i, /\bclarif/i, /\bruling\b/i] },
];

function detectQueryIntent(query) {
  for (const { type, patterns } of INTENT_PATTERNS) {
    for (const pattern of patterns) {
      if (pattern.test(query)) return type;
    }
  }
  return null;
}

function extractQueryRuleRefs(query) {
  const refs = [];
  for (const pattern of RULE_PATTERNS) {
    const regex = new RegExp(pattern.source, pattern.flags);
    let match;
    while ((match = regex.exec(query)) !== null) {
      refs.push(match[1].toLowerCase().replace(/\s+/g, ' '));
    }
  }
  return refs;
}

/**
 * Unified search function composing BM25, synonyms, structural boosting, and optional AI enhancement.
 *
 * @param {object} params
 * @param {BM25Index} params.bm25Index - Pre-built BM25 index
 * @param {Array} params.chunks - All chunks for the category
 * @param {string} params.query - User's question
 * @param {number} params.topK - Number of results to return (default 8)
 * @param {boolean} params.smartSearch - Whether to use Claude query enhancement
 * @param {string} params.apiKey - API key for Claude enhancement
 * @returns {Promise<Array>} - Scored and sorted chunk results
 */
export const searchChunks = async ({
  bm25Index,
  chunks,
  query,
  topK = 8,
  smartSearch = false,
  apiKey = '',
}) => {
  if (!chunks.length || !bm25Index) return [];

  // Build chunk lookup map
  const chunkMap = new Map(chunks.map(c => [c.id, c]));

  // 1. Extract keywords and expand with synonyms
  const queryKeywords = extractKeywords(query);
  const expandedTerms = expandWithSynonyms(queryKeywords);

  // 2. Optional: Claude-powered query enhancement
  if (smartSearch && apiKey) {
    try {
      const enhancedKeywords = await enhanceQuery(query, apiKey);
      const seen = new Set(expandedTerms.map(t => t.term));
      for (const word of enhancedKeywords) {
        if (!seen.has(word)) {
          expandedTerms.push({ term: word, weight: 0.5 });
          seen.add(word);
        }
      }
    } catch {
      // Enhancement failed, continue with synonym expansion only
    }
  }

  // 3. BM25 search
  const bm25Results = bm25Index.search(expandedTerms, topK * 3);

  // 4. Apply bonuses: rule reference matching, exact match, structural boosting
  const queryRuleRefs = extractQueryRuleRefs(query);
  const queryIntent = detectQueryIntent(query);
  const queryLower = query.toLowerCase();

  const scored = bm25Results.map(({ chunkId, score }) => {
    const chunk = chunkMap.get(chunkId);
    if (!chunk) return null;

    let finalScore = score;

    // Rule reference bonus (+5 per matching ref)
    if (queryRuleRefs.length > 0) {
      const chunkText = chunk.text.toLowerCase();
      const chunkRuleRef = (chunk.ruleRef || '').toLowerCase();
      for (const ref of queryRuleRefs) {
        if (chunkText.includes(ref) || chunkRuleRef.includes(ref)) {
          finalScore += 5;
        }
      }
      // Extra bonus if the chunk's primary ruleRef matches
      if (chunk.ruleRef) {
        for (const ref of queryRuleRefs) {
          if (chunkRuleRef.includes(ref)) {
            finalScore *= 2;
            break;
          }
        }
      }
    }

    // Exact query match bonus
    if (chunk.text.toLowerCase().includes(queryLower)) {
      finalScore += 10;
    }

    // Structural boosting: boost chunks whose sectionType matches query intent
    if (queryIntent && chunk.sectionType === queryIntent) {
      finalScore *= 1.5;
    }

    // Heading match bonus: if query keywords appear in the rule reference
    if (chunk.ruleRef) {
      const refLower = chunk.ruleRef.toLowerCase();
      for (const keyword of queryKeywords) {
        if (refLower.includes(keyword)) {
          finalScore *= 1.3;
          break;
        }
      }
    }

    return { ...chunk, score: finalScore };
  }).filter(Boolean);

  // Sort and return top K
  scored.sort((a, b) => b.score - a.score);
  return scored.slice(0, topK);
};
