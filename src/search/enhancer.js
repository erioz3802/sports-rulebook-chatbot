/**
 * Claude-powered query enhancement.
 * Uses Haiku to expand a user's baseball rules question into better search terms.
 */

const ENHANCER_PROMPT = `You are a baseball rules search assistant. Given a user's question about baseball rules, output ONLY a comma-separated list of additional search terms that would help find the relevant rule sections. Include:
- Official rule terminology (e.g., "base on balls" for "walk")
- Relevant rule section numbers if you know them
- Related concepts that would appear near the answer

Output ONLY the comma-separated terms, nothing else. Keep it under 20 terms.`;

/**
 * Call Claude Haiku to expand a query with better search terms.
 * Returns array of expanded keywords, or empty array on failure.
 */
export const enhanceQuery = async (query, apiKey) => {
  try {
    const response = await fetch('https://api.anthropic.com/v1/messages', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'x-api-key': apiKey,
        'anthropic-version': '2023-06-01',
        'anthropic-dangerous-direct-browser-access': 'true'
      },
      body: JSON.stringify({
        model: 'claude-haiku-4-5-20251001',
        max_tokens: 200,
        system: ENHANCER_PROMPT,
        messages: [{ role: 'user', content: query }]
      })
    });

    const data = await response.json();
    const text = data.content?.[0]?.text;
    if (!text) return [];

    // Parse comma-separated terms into individual keywords
    return text
      .toLowerCase()
      .split(/[,\n]+/)
      .map(t => t.trim())
      .filter(t => t.length > 0)
      .flatMap(phrase => phrase.split(/\s+/))
      .filter(w => w.length > 2);
  } catch {
    // Silently fail — enhancer is optional
    return [];
  }
};
