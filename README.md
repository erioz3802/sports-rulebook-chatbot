# ⚾ Baseball Rules Assistant

A RAG-based (Retrieval Augmented Generation) chatbot for baseball rules. Upload rule books (PDF), organize by category, and ask questions.

## Features

- **Category Management**: Create categories like "NFHS", "NCAA", "MLB", etc.
- **PDF Upload**: Upload rule books and case books to each category
- **Smart Search**: Questions search ALL documents in the selected category
- **Persistent Storage**: Everything stored in browser's IndexedDB (survives refreshes)
- **Source Citations**: Shows which documents were used for each answer
- **Secure API Key**: Stored locally, never sent anywhere except Anthropic

## Quick Start

### Prerequisites

- [Node.js](https://nodejs.org/) (version 18 or higher)
- An [Anthropic API key](https://console.anthropic.com/)

### Run Locally

1. **Install dependencies:**
   ```bash
   npm install
   ```

2. **Start the development server:**
   ```bash
   npm run dev
   ```

3. **Open in browser:**
   ```
   http://localhost:5173
   ```

4. **Add your API key:**
   - Click "Settings" in the top right
   - Enter your Anthropic API key
   - Click "Save"

5. **Upload your rule books:**
   - Go to "Manage Documents" tab
   - Create a category (e.g., "NFHS")
   - Upload PDFs to that category

6. **Start asking questions:**
   - Go to "Chat" tab
   - Select your category
   - Ask away!

## Deploy to the Web

### Option 1: Vercel (Recommended - Free)

1. Push your code to GitHub

2. Go to [vercel.com](https://vercel.com) and sign in with GitHub

3. Click "New Project" and import your repository

4. Click "Deploy"

5. Your app will be live at `https://your-project.vercel.app`

### Option 2: Netlify (Free)

1. Build the project:
   ```bash
   npm run build
   ```

2. Go to [netlify.com](https://netlify.com) and sign in

3. Drag and drop the `dist` folder to deploy

4. Your app will be live at `https://your-project.netlify.app`

### Option 3: GitHub Pages (Free)

1. Update `vite.config.js`:
   ```js
   export default defineConfig({
     base: '/your-repo-name/',
     // ... rest of config
   })
   ```

2. Build the project:
   ```bash
   npm run build
   ```

3. Deploy the `dist` folder to GitHub Pages

### Option 4: Any Static Host

The `npm run build` command creates a `dist` folder with static files. Upload these to any web host (AWS S3, Google Cloud Storage, Cloudflare Pages, etc.)

## Usage Guide

### Creating Categories

Categories help organize your rule books. Examples:
- **NFHS** - High school rules
- **NCAA** - College rules
- **MLB** - Professional rules
- **Little League** - Youth rules

### Uploading Documents

Each category can have multiple PDFs:
- NFHS Rules Book
- NFHS Case Book
- NFHS Points of Emphasis
- State-specific supplements

### Asking Questions

When you ask a question:
1. The system searches all documents in the selected category
2. Finds the most relevant sections
3. Sends those sections + your question to Claude
4. Returns an answer with rule citations

### Example Questions

- "Explain the Force Play Slide Rule"
- "What happens if a batter bats out of order?"
- "Is a runner out if hit by a batted ball?"
- "What are the rules for a legal slide?"
- "When is obstruction called?"

## Technical Details

### How RAG Works

1. **PDF Processing**: Extracts text from PDFs using PDF.js
2. **Chunking**: Splits text into ~1500 character chunks with overlap
3. **Search**: Keyword matching + rule reference boosting
4. **Context Building**: Top 8 relevant chunks sent to Claude
5. **Response**: Claude answers based on the provided context

### Storage

- **IndexedDB**: Categories, documents, and text chunks
- **LocalStorage**: Not used (IndexedDB is more robust)
- **No Server**: Everything runs in the browser

### API Usage

Each question uses approximately:
- Input: ~3000-5000 tokens (system prompt + context + question)
- Output: ~500-1000 tokens (answer)

## Troubleshooting

### "API Key not working"
- Make sure you're using an Anthropic API key (starts with `sk-ant-`)
- Check that your API key has credits available
- The key is stored in your browser - try clearing and re-entering

### "PDF upload fails"
- Make sure the PDF is not password-protected
- Try a smaller PDF first to test
- Check browser console for errors

### "No results found"
- Make sure you've uploaded documents to the selected category
- Try different keywords in your question
- Check that the PDF text was extracted properly

## Privacy & Security

- **API Key**: Stored only in your browser's IndexedDB
- **Documents**: Stored only in your browser's IndexedDB
- **No Server**: No data is sent to any server except Anthropic's API
- **CORS**: Uses Anthropic's `anthropic-dangerous-direct-browser-access` header

## License

MIT - Use freely for personal or commercial purposes.

## Credits

Built with:
- [React](https://react.dev/)
- [Vite](https://vitejs.dev/)
- [Tailwind CSS](https://tailwindcss.com/)
- [PDF.js](https://mozilla.github.io/pdf.js/)
- [Claude API](https://anthropic.com/)
