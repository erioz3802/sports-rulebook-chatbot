# 🏆 Sports Rulebook Q&A

A chatbot that answers questions about sports rules based on uploaded rulebooks.

## Quick Deploy to Streamlit Cloud (Free)

### Step 1: Create GitHub Account
1. Go to [github.com](https://github.com)
2. Click "Sign Up" and create a free account

### Step 2: Create a New Repository
1. Click the "+" icon in the top right → "New repository"
2. Name it: `sports-rulebook-qa`
3. Keep it **Public** (required for free Streamlit hosting)
4. Click "Create repository"

### Step 3: Upload Files
1. On your new repository page, click "uploading an existing file"
2. Drag and drop these files:
   - `streamlit_app.py`
   - `requirements.txt`
   - `.gitignore`
3. Click "Commit changes"

### Step 4: Deploy on Streamlit Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click "New app"
3. Connect your GitHub account
4. Select your repository: `sports-rulebook-qa`
5. Main file: `streamlit_app.py`
6. Click "Deploy"

### Step 5: Add Your API Key (Optional)
1. In Streamlit Cloud, go to your app's settings (⚙️)
2. Click "Secrets"
3. Add this:
   ```
   ANTHROPIC_API_KEY = "your-api-key-here"
   ```
4. Click "Save"

### Done! 🎉
Your app will be live at: `https://your-app-name.streamlit.app`

Access it from your phone, computer, anywhere!

---

## Local Development

```bash
pip install -r requirements.txt
python -m streamlit run streamlit_app.py
```

## Features
- 📤 Upload PDF, Word, or text rulebooks
- 🏅 Filter by sport and competition level
- 💬 Ask questions in natural language
- 📋 Get AI-analyzed answers with rule citations
