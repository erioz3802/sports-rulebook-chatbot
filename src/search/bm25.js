import { extractKeywords } from './keywords.js';

// BM25 parameters
const K1 = 1.5;  // term frequency saturation
const B = 0.75;  // length normalization

export class BM25Index {
  constructor() {
    this.docs = [];       // [{id, keywords, length}]
    this.df = new Map();  // term -> number of docs containing it
    this.avgDl = 0;       // average document length
    this.N = 0;           // total number of documents
  }

  build(chunks) {
    this.docs = [];
    this.df = new Map();

    for (const chunk of chunks) {
      const keywords = extractKeywords(chunk.text);
      const termFreqs = new Map();
      for (const word of keywords) {
        termFreqs.set(word, (termFreqs.get(word) || 0) + 1);
      }
      this.docs.push({
        id: chunk.id,
        tf: termFreqs,
        length: keywords.length,
      });

      // count document frequency (each unique term once per doc)
      for (const term of termFreqs.keys()) {
        this.df.set(term, (this.df.get(term) || 0) + 1);
      }
    }

    this.N = this.docs.length;
    this.avgDl = this.N > 0
      ? this.docs.reduce((sum, d) => sum + d.length, 0) / this.N
      : 1;
  }

  // Returns IDF for a term using BM25 variant: log((N - df + 0.5) / (df + 0.5) + 1)
  idf(term) {
    const df = this.df.get(term) || 0;
    return Math.log((this.N - df + 0.5) / (df + 0.5) + 1);
  }

  // Score a single document against query terms
  // queryTerms: array of {term, weight} where weight defaults to 1
  scoreDoc(docIndex, queryTerms) {
    const doc = this.docs[docIndex];
    let score = 0;

    for (const { term, weight } of queryTerms) {
      const tf = doc.tf.get(term) || 0;
      if (tf === 0) continue;

      const idf = this.idf(term);
      const tfNorm = (tf * (K1 + 1)) / (tf + K1 * (1 - B + B * (doc.length / this.avgDl)));
      score += idf * tfNorm * weight;
    }

    return score;
  }

  // Search all docs, return array of {chunkId, score} sorted descending
  search(queryTerms, limit = 20) {
    const results = [];

    for (let i = 0; i < this.docs.length; i++) {
      const score = this.scoreDoc(i, queryTerms);
      if (score > 0) {
        results.push({ chunkId: this.docs[i].id, score });
      }
    }

    results.sort((a, b) => b.score - a.score);
    return results.slice(0, limit);
  }
}
