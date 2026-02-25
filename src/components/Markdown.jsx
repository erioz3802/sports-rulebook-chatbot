import React from 'react';

function parseInline(text) {
  const elements = [];
  let remaining = text;
  let key = 0;

  while (remaining.length > 0) {
    // Bold: **text**
    let match = remaining.match(/^\*\*(.+?)\*\*/);
    if (match) {
      elements.push(<strong key={key++}>{parseInline(match[1])}</strong>);
      remaining = remaining.slice(match[0].length);
      continue;
    }

    // Italic: *text*
    match = remaining.match(/^\*(.+?)\*/);
    if (match) {
      elements.push(<em key={key++}>{parseInline(match[1])}</em>);
      remaining = remaining.slice(match[0].length);
      continue;
    }

    // Inline code: `text`
    match = remaining.match(/^`([^`]+)`/);
    if (match) {
      elements.push(
        <code key={key++} className="bg-gray-100 text-red-700 px-1.5 py-0.5 rounded text-sm font-mono">
          {match[1]}
        </code>
      );
      remaining = remaining.slice(match[0].length);
      continue;
    }

    // Plain text up to next special character
    match = remaining.match(/^[^*`]+/);
    if (match) {
      elements.push(match[0]);
      remaining = remaining.slice(match[0].length);
      continue;
    }

    // Lone special character
    elements.push(remaining[0]);
    remaining = remaining.slice(1);
  }

  return elements.length === 1 ? elements[0] : elements;
}

export default function Markdown({ content }) {
  if (!content) return null;

  const lines = content.split('\n');
  const blocks = [];
  let i = 0;
  let key = 0;

  while (i < lines.length) {
    const line = lines[i];

    // Fenced code block
    if (line.trimStart().startsWith('```')) {
      const codeLines = [];
      i++;
      while (i < lines.length && !lines[i].trimStart().startsWith('```')) {
        codeLines.push(lines[i]);
        i++;
      }
      i++; // skip closing ```
      blocks.push(
        <pre key={key++} className="bg-gray-900 text-green-300 rounded-lg p-4 overflow-x-auto my-2 text-sm font-mono">
          <code>{codeLines.join('\n')}</code>
        </pre>
      );
      continue;
    }

    // Headers
    const headerMatch = line.match(/^(#{1,6})\s+(.+)/);
    if (headerMatch) {
      const level = headerMatch[1].length;
      const text = headerMatch[2];
      const Tag = `h${level}`;
      const sizes = {
        1: 'text-2xl font-bold mt-4 mb-2',
        2: 'text-xl font-bold mt-3 mb-2',
        3: 'text-lg font-semibold mt-3 mb-1',
        4: 'text-base font-semibold mt-2 mb-1',
        5: 'text-sm font-semibold mt-2 mb-1',
        6: 'text-sm font-medium mt-2 mb-1',
      };
      blocks.push(
        <Tag key={key++} className={sizes[level]}>
          {parseInline(text)}
        </Tag>
      );
      i++;
      continue;
    }

    // Blockquote
    if (line.startsWith('> ')) {
      const quoteLines = [];
      while (i < lines.length && lines[i].startsWith('> ')) {
        quoteLines.push(lines[i].slice(2));
        i++;
      }
      blocks.push(
        <blockquote key={key++} className="border-l-4 border-green-400 pl-4 my-2 text-gray-600 italic">
          {quoteLines.map((ql, qi) => (
            <p key={qi}>{parseInline(ql)}</p>
          ))}
        </blockquote>
      );
      continue;
    }

    // Unordered list
    if (/^[\s]*[-*+]\s/.test(line)) {
      const items = [];
      while (i < lines.length && /^[\s]*[-*+]\s/.test(lines[i])) {
        items.push(lines[i].replace(/^[\s]*[-*+]\s/, ''));
        i++;
      }
      blocks.push(
        <ul key={key++} className="list-disc list-inside my-2 space-y-1 ml-2">
          {items.map((item, ii) => (
            <li key={ii}>{parseInline(item)}</li>
          ))}
        </ul>
      );
      continue;
    }

    // Ordered list
    if (/^[\s]*\d+[.)]\s/.test(line)) {
      const items = [];
      while (i < lines.length && /^[\s]*\d+[.)]\s/.test(lines[i])) {
        items.push(lines[i].replace(/^[\s]*\d+[.)]\s/, ''));
        i++;
      }
      blocks.push(
        <ol key={key++} className="list-decimal list-inside my-2 space-y-1 ml-2">
          {items.map((item, ii) => (
            <li key={ii}>{parseInline(item)}</li>
          ))}
        </ol>
      );
      continue;
    }

    // Horizontal rule
    if (/^[-*_]{3,}\s*$/.test(line)) {
      blocks.push(<hr key={key++} className="my-3 border-gray-300" />);
      i++;
      continue;
    }

    // Empty line
    if (line.trim() === '') {
      i++;
      continue;
    }

    // Regular paragraph — collect consecutive non-empty, non-special lines
    const paraLines = [];
    while (
      i < lines.length &&
      lines[i].trim() !== '' &&
      !lines[i].match(/^#{1,6}\s/) &&
      !lines[i].startsWith('> ') &&
      !lines[i].trimStart().startsWith('```') &&
      !/^[\s]*[-*+]\s/.test(lines[i]) &&
      !/^[\s]*\d+[.)]\s/.test(lines[i]) &&
      !/^[-*_]{3,}\s*$/.test(lines[i])
    ) {
      paraLines.push(lines[i]);
      i++;
    }
    if (paraLines.length > 0) {
      blocks.push(
        <p key={key++} className="my-1">
          {paraLines.map((pl, pi) => (
            <React.Fragment key={pi}>
              {pi > 0 && <br />}
              {parseInline(pl)}
            </React.Fragment>
          ))}
        </p>
      );
    }
  }

  return <div className="markdown-content">{blocks}</div>;
}
