// Baseball-specific synonym groups
// Each group is an array of terms that should be treated as equivalent
const SYNONYM_GROUPS = [
  ['balk', 'illegal pitch'],
  ['walk', 'base on balls', 'bb', 'four balls'],
  ['tag up', 'retouch', 'retouching'],
  ['hit by pitch', 'hbp', 'hit batter', 'hit batsman'],
  ['strikeout', 'strike out', 'struck out', 'punchout'],
  ['ground rule double', 'automatic double', 'book rule double'],
  ['fair ball', 'fair territory', 'fair ground'],
  ['foul ball', 'foul territory', 'foul ground'],
  ['infield fly', 'infield fly rule'],
  ['designated hitter', 'dh'],
  ['extra hitter', 'eh'],
  ['courtesy runner', 'cr', 'special pinch runner'],
  ['obstruction', 'obstructing'],
  ['interference', 'interfering', 'interfere'],
  ['steal', 'stolen base', 'stealing'],
  ['bunt', 'bunting', 'bunted'],
  ['sacrifice', 'sac bunt', 'sac fly', 'sacrifice fly', 'sacrifice bunt'],
  ['force play', 'force out', 'forced'],
  ['tag play', 'tag out', 'tagged'],
  ['dead ball', 'ball is dead', 'dead-ball'],
  ['live ball', 'ball is live', 'ball in play'],
  ['pitcher', 'pitching', 'mound'],
  ['catcher', 'catching', 'behind the plate'],
  ['umpire', 'ump', 'official', 'plate umpire', 'base umpire'],
  ['ejection', 'ejected', 'thrown out', 'tossed'],
  ['appeal', 'appeal play', 'appealing'],
  ['run scored', 'scoring', 'scores', 'crossed the plate'],
  ['batting order', 'lineup', 'batting lineup'],
  ['substitute', 'substitution', 'sub', 'replacement'],
  ['warmup', 'warm-up', 'warm up', 'warmup pitches'],
  ['time out', 'timeout', 'time'],
  ['double play', 'dp', 'twin killing'],
  ['triple play', 'tp'],
  ['earned run', 'era', 'earned run average'],
  ['error', 'errors', 'fielding error'],
  ['passed ball', 'pb'],
  ['wild pitch', 'wp'],
];

// Build lookup map: term -> Set of synonyms (not including the term itself)
const synonymMap = new Map();

for (const group of SYNONYM_GROUPS) {
  for (const term of group) {
    const lower = term.toLowerCase();
    if (!synonymMap.has(lower)) {
      synonymMap.set(lower, new Set());
    }
    for (const other of group) {
      const otherLower = other.toLowerCase();
      if (otherLower !== lower) {
        synonymMap.get(lower).add(otherLower);
      }
    }
  }
}

/**
 * Expand a query string with baseball synonyms.
 * Returns array of {term, weight} suitable for BM25 search.
 * Direct query terms get weight=1, synonyms get weight=0.5.
 */
export const expandWithSynonyms = (queryKeywords) => {
  const terms = [];
  const seen = new Set();

  // Add original keywords at full weight
  for (const word of queryKeywords) {
    if (!seen.has(word)) {
      terms.push({ term: word, weight: 1 });
      seen.add(word);
    }
  }

  // Check for multi-word synonym matches in the original query
  const queryText = queryKeywords.join(' ');
  for (const [phrase, syns] of synonymMap.entries()) {
    if (phrase.includes(' ') && queryText.includes(phrase)) {
      for (const syn of syns) {
        // Add each word of the synonym
        for (const word of syn.split(/\s+/)) {
          if (word.length > 2 && !seen.has(word)) {
            terms.push({ term: word, weight: 0.5 });
            seen.add(word);
          }
        }
      }
    }
  }

  // Check single-word synonyms
  for (const word of queryKeywords) {
    const syns = synonymMap.get(word);
    if (syns) {
      for (const syn of syns) {
        for (const synWord of syn.split(/\s+/)) {
          if (synWord.length > 2 && !seen.has(synWord)) {
            terms.push({ term: synWord, weight: 0.5 });
            seen.add(synWord);
          }
        }
      }
    }
  }

  return terms;
};
