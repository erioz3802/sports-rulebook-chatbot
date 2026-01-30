# ⚾ Baseball Rules Assistant (Google Drive Edition)

A RAG-based baseball rules chatbot that syncs your data across all devices using Google Drive.

## Features

- **☁️ Cloud Sync**: All data stored in YOUR Google Drive
- **📱 Multi-Device**: Access from PC, phone, tablet - same data everywhere
- **🔐 Secure**: Only you can access your data (via Google Sign-In)
- **📚 Categories**: Organize rule books (NFHS, NCAA, etc.)
- **🔍 Smart Search**: Searches all documents in selected category
- **💾 Persistent**: Upload PDFs once, use forever

## How It Works

```
Your Devices                          Your Google Drive
┌────────────────┐                   ┌─────────────────────────┐
│  💻 PC         │──┐                │  📁 BaseballRulesAssistant │
│  📱 Phone      │──┼── Sign In ───▶│    ├── config.json      │
│  📱 Tablet     │──┘    with       │    ├── chunks.json      │
└────────────────┘      Google      │    └── 📄 Your PDFs     │
                                    └─────────────────────────┘
```

## Quick Start

### 1. Set Up Google Cloud (One-Time, ~5 minutes)

1. Go to [Google Cloud Console](https://console.cloud.google.com)

2. **Create a Project:**
   - Click "Select a project" → "New Project"
   - Name it "Baseball Rules Assistant"
   - Click "Create"

3. **Enable Google Drive API:**
   - Go to "APIs & Services" → "Library"
   - Search for "Google Drive API"
   - Click on it → Click "Enable"

4. **Configure OAuth Consent Screen:**
   - Go to "APIs & Services" → "OAuth consent screen"
   - Choose "External" → Click "Create"
   - Fill in:
     - App name: `Baseball Rules Assistant`
     - User support email: your email
     - Developer contact: your email
   - Click "Save and Continue"
   - Skip "Scopes" → Click "Save and Continue"
   - Add yourself as a test user (your Gmail address)
   - Click "Save and Continue" → "Back to Dashboard"

5. **Create OAuth Credentials:**
   - Go to "APIs & Services" → "Credentials"
   - Click "Create Credentials" → "OAuth client ID"
   - Application type: **Web application**
   - Name: `Baseball Rules Web Client`
   - **Authorized JavaScript origins:**
     - `http://localhost:5173` (for local development)
     - `https://your-app.onrender.com` (your deployed URL)
   - Click "Create"
   - **Copy the Client ID** (looks like: `123456789-abc123.apps.googleusercontent.com`)

### 2. Run the App

```bash
# Install dependencies
npm install

# Start development server
npm run dev

# Open http://localhost:5173
```

### 3. First-Time Setup in App

1. Paste your Google Client ID when prompted
2. Click "Sign in with Google"
3. Grant permission to access Google Drive
4. Add your Anthropic API key in Settings
5. Create categories and upload PDFs
6. Start asking questions!

## Deploying to Render

1. Push your code to GitHub

2. Go to [render.com](https://render.com) → New → Static Site

3. Configure:
   | Field | Value |
   |-------|-------|
   | Build Command | `npm install && npm run build` |
   | Publish Directory | `dist` |

4. **Important:** After deploying, add your Render URL to Google Cloud Console:
   - Go to "APIs & Services" → "Credentials"
   - Edit your OAuth client
   - Add `https://your-app.onrender.com` to "Authorized JavaScript origins"

## What's Stored Where

| Data | Location | Synced? |
|------|----------|---------|
| Google Client ID | Browser localStorage | ❌ Per browser |
| Auth Token | Browser (auto-refreshes) | ❌ Per browser |
| Anthropic API Key | Google Drive | ✅ All devices |
| Categories | Google Drive | ✅ All devices |
| Documents/Chunks | Google Drive | ✅ All devices |
| PDF Files | Google Drive | ✅ All devices |

## Troubleshooting

### "Sign in with Google" doesn't work

1. Check that your Client ID is correct
2. Make sure your current URL is in "Authorized JavaScript origins"
3. Make sure you added yourself as a test user in OAuth consent screen

### "Access blocked" error

Your app is in "Testing" mode. You need to:
1. Add your Google account as a test user, OR
2. Publish the app (requires Google verification for public apps)

### Token expired

Just click "Sign in with Google" again - it's one click if you're already signed into Google.

### Data not syncing

Check your internet connection. The app saves to Google Drive after each change.

## Privacy

- **Your data stays in YOUR Google Drive** - nobody else can see it
- **No server storage** - the app is just static files
- **API keys in your Drive** - not stored on any server
- **Google handles auth** - we never see your Google password

## Cost

**$0** - Everything is free:
- Google Drive: 15GB free (your PDFs probably use < 200MB)
- Google Drive API: Free, no limits for personal use
- Render Static Site: Free tier
- Anthropic API: Pay per use (a few cents per conversation)

## License

MIT
