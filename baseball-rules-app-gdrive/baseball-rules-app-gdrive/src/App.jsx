import React, { useState, useEffect, useRef, useCallback } from 'react';
import * as pdfjsLib from 'pdfjs-dist';

// Set up PDF.js worker for v5+
pdfjsLib.GlobalWorkerOptions.workerSrc = new URL(
  'pdfjs-dist/build/pdf.worker.min.mjs',
  import.meta.url
).toString();

// ============ Google Drive API Helper ============
const SCOPES = 'https://www.googleapis.com/auth/drive.file';
const APP_FOLDER_NAME = 'BaseballRulesAssistant';
const CONFIG_FILE_NAME = 'config.json';
const CHUNKS_FILE_NAME = 'chunks.json';

class GoogleDriveService {
  constructor() {
    this.tokenClient = null;
    this.accessToken = null;
    this.appFolderId = null;
  }

  async init(clientId, onAuthChange) {
    return new Promise((resolve) => {
      // Load Google Identity Services
      const script = document.createElement('script');
      script.src = 'https://accounts.google.com/gsi/client';
      script.onload = () => {
        this.tokenClient = window.google.accounts.oauth2.initTokenClient({
          client_id: clientId,
          scope: SCOPES,
          callback: async (response) => {
            if (response.access_token) {
              this.accessToken = response.access_token;
              localStorage.setItem('gdriveToken', response.access_token);
              await this.ensureAppFolder();
              onAuthChange(true);
            }
          },
        });
        
        // Check for existing token
        const savedToken = localStorage.getItem('gdriveToken');
        if (savedToken) {
          this.accessToken = savedToken;
          this.validateToken().then(valid => {
            if (valid) {
              this.ensureAppFolder().then(() => onAuthChange(true));
            } else {
              localStorage.removeItem('gdriveToken');
              onAuthChange(false);
            }
          });
        }
        resolve();
      };
      document.body.appendChild(script);
    });
  }

  async validateToken() {
    if (!this.accessToken) return false;
    try {
      const response = await fetch('https://www.googleapis.com/drive/v3/about?fields=user', {
        headers: { Authorization: `Bearer ${this.accessToken}` }
      });
      return response.ok;
    } catch {
      return false;
    }
  }

  signIn() {
    this.tokenClient.requestAccessToken({ prompt: 'consent' });
  }

  signOut(onAuthChange) {
    if (this.accessToken) {
      window.google.accounts.oauth2.revoke(this.accessToken);
      this.accessToken = null;
      this.appFolderId = null;
      localStorage.removeItem('gdriveToken');
      onAuthChange(false);
    }
  }

  async ensureAppFolder() {
    // Check if folder exists
    const searchResponse = await fetch(
      `https://www.googleapis.com/drive/v3/files?q=name='${APP_FOLDER_NAME}' and mimeType='application/vnd.google-apps.folder' and trashed=false`,
      { headers: { Authorization: `Bearer ${this.accessToken}` } }
    );
    const searchData = await searchResponse.json();
    
    if (searchData.files && searchData.files.length > 0) {
      this.appFolderId = searchData.files[0].id;
    } else {
      // Create folder
      const createResponse = await fetch('https://www.googleapis.com/drive/v3/files', {
        method: 'POST',
        headers: {
          Authorization: `Bearer ${this.accessToken}`,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          name: APP_FOLDER_NAME,
          mimeType: 'application/vnd.google-apps.folder'
        })
      });
      const createData = await createResponse.json();
      this.appFolderId = createData.id;
    }
    return this.appFolderId;
  }

  async findFile(fileName) {
    const response = await fetch(
      `https://www.googleapis.com/drive/v3/files?q=name='${fileName}' and '${this.appFolderId}' in parents and trashed=false`,
      { headers: { Authorization: `Bearer ${this.accessToken}` } }
    );
    const data = await response.json();
    return data.files && data.files.length > 0 ? data.files[0] : null;
  }

  async readJsonFile(fileName) {
    const file = await this.findFile(fileName);
    if (!file) return null;
    
    const response = await fetch(
      `https://www.googleapis.com/drive/v3/files/${file.id}?alt=media`,
      { headers: { Authorization: `Bearer ${this.accessToken}` } }
    );
    return response.json();
  }

  async writeJsonFile(fileName, data) {
    const file = await this.findFile(fileName);
    const content = JSON.stringify(data);
    const blob = new Blob([content], { type: 'application/json' });
    
    if (file) {
      // Update existing file
      await fetch(`https://www.googleapis.com/upload/drive/v3/files/${file.id}?uploadType=media`, {
        method: 'PATCH',
        headers: { Authorization: `Bearer ${this.accessToken}` },
        body: blob
      });
    } else {
      // Create new file
      const metadata = {
        name: fileName,
        parents: [this.appFolderId]
      };
      
      const form = new FormData();
      form.append('metadata', new Blob([JSON.stringify(metadata)], { type: 'application/json' }));
      form.append('file', blob);
      
      await fetch('https://www.googleapis.com/upload/drive/v3/files?uploadType=multipart', {
        method: 'POST',
        headers: { Authorization: `Bearer ${this.accessToken}` },
        body: form
      });
    }
  }

  async uploadPdf(file, categoryId) {
    const metadata = {
      name: `${categoryId}_${Date.now()}_${file.name}`,
      parents: [this.appFolderId]
    };
    
    const form = new FormData();
    form.append('metadata', new Blob([JSON.stringify(metadata)], { type: 'application/json' }));
    form.append('file', file);
    
    const response = await fetch('https://www.googleapis.com/upload/drive/v3/files?uploadType=multipart', {
      method: 'POST',
      headers: { Authorization: `Bearer ${this.accessToken}` },
      body: form
    });
    
    return response.json();
  }

  async deletePdfFile(fileId) {
    await fetch(`https://www.googleapis.com/drive/v3/files/${fileId}`, {
      method: 'DELETE',
      headers: { Authorization: `Bearer ${this.accessToken}` }
    });
  }

  async downloadPdf(fileId) {
    const response = await fetch(
      `https://www.googleapis.com/drive/v3/files/${fileId}?alt=media`,
      { headers: { Authorization: `Bearer ${this.accessToken}` } }
    );
    return response.arrayBuffer();
  }
}

const driveService = new GoogleDriveService();

// ============ Text Processing ============
const chunkText = (text, chunkSize = 1500, overlap = 200) => {
  const chunks = [];
  let start = 0;
  while (start < text.length) {
    const end = Math.min(start + chunkSize, text.length);
    chunks.push(text.slice(start, end));
    start = end - overlap;
    if (start + overlap >= text.length) break;
  }
  return chunks;
};

const extractKeywords = (text) => {
  const stopWords = new Set(['the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might',
    'must', 'shall', 'can', 'need', 'dare', 'ought', 'used', 'to', 'of', 'in', 'for', 'on', 'with',
    'at', 'by', 'from', 'as', 'into', 'through', 'during', 'before', 'after', 'above', 'below',
    'between', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where',
    'why', 'how', 'all', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
    'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 'just', 'and', 'but', 'if', 'or',
    'because', 'until', 'while', 'this', 'that', 'these', 'those', 'what', 'which', 'who']);
  
  return text.toLowerCase()
    .replace(/[^\w\s-]/g, ' ')
    .split(/\s+/)
    .filter(word => word.length > 2 && !stopWords.has(word));
};

const searchChunks = (chunks, query, topK = 8) => {
  const queryKeywords = new Set(extractKeywords(query));
  
  const scored = chunks.map(chunk => {
    const chunkKeywords = extractKeywords(chunk.text);
    let score = 0;
    
    chunkKeywords.forEach(word => {
      if (queryKeywords.has(word)) score += 1;
    });
    
    const rulePattern = /rule\s*\d+[-.]?\d*/gi;
    const queryRules = query.match(rulePattern) || [];
    const chunkRules = chunk.text.match(rulePattern) || [];
    queryRules.forEach(rule => {
      if (chunkRules.some(cr => cr.toLowerCase().includes(rule.toLowerCase()))) {
        score += 5;
      }
    });
    
    if (chunk.text.toLowerCase().includes(query.toLowerCase())) {
      score += 10;
    }
    
    return { ...chunk, score };
  });
  
  return scored
    .filter(c => c.score > 0)
    .sort((a, b) => b.score - a.score)
    .slice(0, topK);
};

// ============ Main Component ============
export default function App() {
  // Google Client ID - Users will set this
  const [clientId, setClientId] = useState(() => localStorage.getItem('googleClientId') || '');
  const [clientIdInput, setClientIdInput] = useState('');
  const [showSetup, setShowSetup] = useState(false);
  
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [isInitialized, setIsInitialized] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  
  const [apiKey, setApiKey] = useState('');
  const [apiKeyInput, setApiKeyInput] = useState('');
  const [categories, setCategories] = useState([]);
  const [documents, setDocuments] = useState([]);
  const [chunks, setChunks] = useState([]);
  const [selectedCategory, setSelectedCategory] = useState(null);
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isSending, setIsSending] = useState(false);
  const [activeTab, setActiveTab] = useState('chat');
  const [uploadProgress, setUploadProgress] = useState(null);
  const [newCategoryName, setNewCategoryName] = useState('');
  const [showSettings, setShowSettings] = useState(false);
  const [saveStatus, setSaveStatus] = useState('');
  const messagesEndRef = useRef(null);

  // Initialize Google Drive
  useEffect(() => {
    if (clientId) {
      driveService.init(clientId, (authenticated) => {
        setIsAuthenticated(authenticated);
        setIsInitialized(true);
        if (authenticated) {
          loadDataFromDrive();
        } else {
          setIsLoading(false);
        }
      });
    } else {
      setIsLoading(false);
    }
  }, [clientId]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const loadDataFromDrive = async () => {
    setIsLoading(true);
    try {
      const config = await driveService.readJsonFile(CONFIG_FILE_NAME);
      if (config) {
        setApiKey(config.apiKey || '');
        setCategories(config.categories || []);
        setDocuments(config.documents || []);
      }
      
      const chunksData = await driveService.readJsonFile(CHUNKS_FILE_NAME);
      if (chunksData) {
        setChunks(chunksData);
      }
    } catch (error) {
      console.error('Error loading from Drive:', error);
    }
    setIsLoading(false);
  };

  const saveConfigToDrive = useCallback(async (newApiKey, newCategories, newDocuments) => {
    setSaveStatus('Saving...');
    try {
      await driveService.writeJsonFile(CONFIG_FILE_NAME, {
        apiKey: newApiKey,
        categories: newCategories,
        documents: newDocuments
      });
      setSaveStatus('Saved ✓');
      setTimeout(() => setSaveStatus(''), 2000);
    } catch (error) {
      console.error('Error saving config:', error);
      setSaveStatus('Save failed');
    }
  }, []);

  const saveChunksToDrive = useCallback(async (newChunks) => {
    try {
      await driveService.writeJsonFile(CHUNKS_FILE_NAME, newChunks);
    } catch (error) {
      console.error('Error saving chunks:', error);
    }
  }, []);

  const handleSignIn = () => {
    driveService.signIn();
  };

  const handleSignOut = () => {
    driveService.signOut(setIsAuthenticated);
    setApiKey('');
    setCategories([]);
    setDocuments([]);
    setChunks([]);
    setMessages([]);
  };

  const saveClientId = () => {
    if (clientIdInput.trim()) {
      localStorage.setItem('googleClientId', clientIdInput.trim());
      setClientId(clientIdInput.trim());
      setClientIdInput('');
      setShowSetup(false);
    }
  };

  const saveApiKey = async () => {
    if (apiKeyInput.trim()) {
      const newApiKey = apiKeyInput.trim();
      setApiKey(newApiKey);
      setApiKeyInput('');
      setShowSettings(false);
      await saveConfigToDrive(newApiKey, categories, documents);
    }
  };

  const clearApiKey = async () => {
    setApiKey('');
    await saveConfigToDrive('', categories, documents);
  };

  const addCategory = async () => {
    if (!newCategoryName.trim()) return;
    
    const category = {
      id: Date.now().toString(),
      name: newCategoryName.trim(),
      createdAt: new Date().toISOString()
    };
    
    const newCategories = [...categories, category];
    setCategories(newCategories);
    setNewCategoryName('');
    await saveConfigToDrive(apiKey, newCategories, documents);
  };

  const deleteCategory = async (categoryId) => {
    if (!confirm('Delete this category and all its documents?')) return;
    
    // Delete PDF files from Drive
    const docsToDelete = documents.filter(d => d.categoryId === categoryId);
    for (const doc of docsToDelete) {
      if (doc.driveFileId) {
        await driveService.deletePdfFile(doc.driveFileId);
      }
    }
    
    const newCategories = categories.filter(c => c.id !== categoryId);
    const newDocuments = documents.filter(d => d.categoryId !== categoryId);
    const newChunks = chunks.filter(c => c.categoryId !== categoryId);
    
    setCategories(newCategories);
    setDocuments(newDocuments);
    setChunks(newChunks);
    
    if (selectedCategory === categoryId) setSelectedCategory(null);
    
    await saveConfigToDrive(apiKey, newCategories, newDocuments);
    await saveChunksToDrive(newChunks);
  };

  const extractTextFromPDF = async (arrayBuffer) => {
    const typedArray = new Uint8Array(arrayBuffer);
    const pdf = await pdfjsLib.getDocument(typedArray).promise;
    let fullText = '';
    
    for (let i = 1; i <= pdf.numPages; i++) {
      setUploadProgress(`Extracting page ${i} of ${pdf.numPages}...`);
      const page = await pdf.getPage(i);
      const textContent = await page.getTextContent();
      const pageText = textContent.items.map(item => item.str).join(' ');
      fullText += `\n[Page ${i}]\n${pageText}\n`;
    }
    
    return { text: fullText, pageCount: pdf.numPages };
  };

  const uploadDocument = async (file, categoryId) => {
    if (!file || !categoryId) return;
    
    setUploadProgress('Uploading to Google Drive...');
    
    try {
      // Upload PDF to Drive
      const driveFile = await driveService.uploadPdf(file, categoryId);
      
      // Read file for text extraction
      const arrayBuffer = await file.arrayBuffer();
      const { text, pageCount } = await extractTextFromPDF(arrayBuffer);
      
      const document = {
        id: Date.now().toString(),
        categoryId,
        name: file.name,
        driveFileId: driveFile.id,
        uploadedAt: new Date().toISOString(),
        pageCount
      };
      
      setUploadProgress('Processing text chunks...');
      const textChunks = chunkText(text, 1500, 200);
      
      const newChunks = textChunks.map((chunkText, i) => ({
        id: `${document.id}_${i}`,
        documentId: document.id,
        categoryId,
        text: chunkText,
        index: i
      }));
      
      const updatedDocuments = [...documents, document];
      const updatedChunks = [...chunks, ...newChunks];
      
      setDocuments(updatedDocuments);
      setChunks(updatedChunks);
      
      setUploadProgress('Saving to cloud...');
      await saveConfigToDrive(apiKey, categories, updatedDocuments);
      await saveChunksToDrive(updatedChunks);
      
      setUploadProgress(null);
    } catch (error) {
      console.error('Error uploading document:', error);
      setUploadProgress(`Error: ${error.message}`);
      setTimeout(() => setUploadProgress(null), 3000);
    }
  };

  const deleteDocument = async (documentId) => {
    if (!confirm('Delete this document?')) return;
    
    const doc = documents.find(d => d.id === documentId);
    if (doc?.driveFileId) {
      await driveService.deletePdfFile(doc.driveFileId);
    }
    
    const newDocuments = documents.filter(d => d.id !== documentId);
    const newChunks = chunks.filter(c => c.documentId !== documentId);
    
    setDocuments(newDocuments);
    setChunks(newChunks);
    
    await saveConfigToDrive(apiKey, categories, newDocuments);
    await saveChunksToDrive(newChunks);
  };

  const sendMessage = async () => {
    if (!input.trim() || isSending || !selectedCategory || !apiKey) return;
    
    const userMessage = { role: 'user', content: input.trim() };
    const newMessages = [...messages, userMessage];
    setMessages(newMessages);
    setInput('');
    setIsSending(true);
    
    try {
      const categoryChunks = chunks.filter(c => c.categoryId === selectedCategory);
      
      if (categoryChunks.length === 0) {
        setMessages([...newMessages, {
          role: 'assistant',
          content: "I don't have any documents loaded for this category yet. Please upload some PDFs first in the Manage tab."
        }]);
        setIsSending(false);
        return;
      }
      
      const relevantChunks = searchChunks(categoryChunks, input.trim(), 8);
      
      const context = relevantChunks.map((chunk) => {
        const doc = documents.find(d => d.id === chunk.documentId);
        return `[Source: ${doc?.name || 'Unknown'}]\n${chunk.text}`;
      }).join('\n\n---\n\n');
      
      const categoryName = categories.find(c => c.id === selectedCategory)?.name || 'Unknown';
      
      const systemPrompt = `You are an expert baseball rules advisor specializing in ${categoryName} rules. 
      
You have access to the following reference material from the user's uploaded rule books:

${context}

Guidelines:
1. Answer questions based on the provided reference material
2. Always cite specific rules when possible (e.g., "Rule 8-4-2b")
3. If the answer isn't in the provided material, say so honestly
4. Be conversational but precise - like a veteran umpire mentor
5. Use examples to illustrate complex situations
6. Format responses clearly with headers and bullets for complex answers
7. Explain the reasoning/purpose behind rules when helpful`;

      const response = await fetch('https://api.anthropic.com/v1/messages', {
        method: 'POST',
        headers: { 
          'Content-Type': 'application/json',
          'x-api-key': apiKey,
          'anthropic-version': '2023-06-01',
          'anthropic-dangerous-direct-browser-access': 'true'
        },
        body: JSON.stringify({
          model: 'claude-sonnet-4-20250514',
          max_tokens: 2000,
          system: systemPrompt,
          messages: newMessages.slice(-10).map(m => ({ role: m.role, content: m.content }))
        })
      });

      const data = await response.json();
      
      if (data.content?.[0]?.text) {
        const sourceDocs = [...new Set(relevantChunks.map(c => {
          const doc = documents.find(d => d.id === c.documentId);
          return doc?.name;
        }).filter(Boolean))];
        
        setMessages([...newMessages, { 
          role: 'assistant', 
          content: data.content[0].text,
          sources: sourceDocs
        }]);
      } else if (data.error) {
        throw new Error(data.error.message);
      }
    } catch (error) {
      setMessages([...newMessages, { 
        role: 'assistant', 
        content: `Error: ${error.message}. Please check your API key and try again.`
      }]);
    } finally {
      setIsSending(false);
    }
  };

  const clearChat = () => {
    setMessages([]);
  };

  const getCategoryDocs = (categoryId) => documents.filter(d => d.categoryId === categoryId);
  const getCategoryChunkCount = (categoryId) => chunks.filter(c => c.categoryId === categoryId).length;

  // ============ RENDER ============

  // Setup screen - no client ID configured
  if (!clientId) {
    return (
      <div className="min-h-screen bg-gradient-to-b from-green-50 to-white flex items-center justify-center p-4">
        <div className="bg-white rounded-2xl shadow-xl p-8 max-w-lg w-full">
          <div className="text-center mb-6">
            <span className="text-6xl">⚾</span>
            <h1 className="text-2xl font-bold mt-4">Baseball Rules Assistant</h1>
            <p className="text-gray-600 mt-2">First-time setup required</p>
          </div>
          
          <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 mb-6">
            <h3 className="font-semibold text-blue-800 mb-2">Setup Instructions:</h3>
            <ol className="text-sm text-blue-900 space-y-2 list-decimal list-inside">
              <li>Go to <a href="https://console.cloud.google.com" target="_blank" rel="noopener noreferrer" className="underline">Google Cloud Console</a></li>
              <li>Create a new project (or select existing)</li>
              <li>Enable the <strong>Google Drive API</strong></li>
              <li>Go to Credentials → Create Credentials → OAuth Client ID</li>
              <li>Application type: <strong>Web application</strong></li>
              <li>Add your app URL to Authorized JavaScript origins</li>
              <li>Copy the Client ID and paste below</li>
            </ol>
          </div>
          
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Google OAuth Client ID
              </label>
              <input
                type="text"
                value={clientIdInput}
                onChange={(e) => setClientIdInput(e.target.value)}
                placeholder="123456789-abc123.apps.googleusercontent.com"
                className="w-full border border-gray-300 rounded-lg px-4 py-2 focus:outline-none focus:ring-2 focus:ring-green-500"
              />
            </div>
            <button
              onClick={saveClientId}
              disabled={!clientIdInput.trim()}
              className="w-full bg-green-600 hover:bg-green-700 disabled:bg-gray-300 text-white py-3 rounded-lg font-medium transition-colors"
            >
              Save & Continue
            </button>
          </div>
        </div>
      </div>
    );
  }

  // Loading screen
  if (isLoading) {
    return (
      <div className="min-h-screen bg-gradient-to-b from-green-50 to-white flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin text-6xl mb-4">⚾</div>
          <p className="text-gray-600">Loading...</p>
        </div>
      </div>
    );
  }

  // Sign in screen
  if (!isAuthenticated) {
    return (
      <div className="min-h-screen bg-gradient-to-b from-green-50 to-white flex items-center justify-center p-4">
        <div className="bg-white rounded-2xl shadow-xl p-8 max-w-md w-full text-center">
          <span className="text-6xl">⚾</span>
          <h1 className="text-2xl font-bold mt-4">Baseball Rules Assistant</h1>
          <p className="text-gray-600 mt-2 mb-6">Sign in to access your rule books from any device</p>
          
          <button
            onClick={handleSignIn}
            className="w-full bg-white border-2 border-gray-300 hover:border-gray-400 py-3 rounded-lg font-medium transition-colors flex items-center justify-center gap-3"
          >
            <svg className="w-5 h-5" viewBox="0 0 24 24">
              <path fill="#4285F4" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"/>
              <path fill="#34A853" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"/>
              <path fill="#FBBC05" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"/>
              <path fill="#EA4335" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"/>
            </svg>
            Sign in with Google
          </button>
          
          <button
            onClick={() => {
              localStorage.removeItem('googleClientId');
              setClientId('');
            }}
            className="mt-4 text-sm text-gray-500 hover:text-gray-700"
          >
            Change Google Client ID
          </button>
        </div>
      </div>
    );
  }

  // Main app
  return (
    <div className="flex flex-col h-screen bg-gray-100">
      {/* Header */}
      <div className="bg-gradient-to-r from-green-700 to-green-600 text-white p-4 shadow-lg">
        <div className="max-w-6xl mx-auto flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold flex items-center gap-2">
              ⚾ Baseball Rules Assistant
            </h1>
            <p className="text-green-200 text-sm mt-1">
              Synced to Google Drive
              {saveStatus && <span className="ml-2">• {saveStatus}</span>}
            </p>
          </div>
          <div className="flex items-center gap-3">
            <button
              onClick={() => setShowSettings(!showSettings)}
              className="bg-green-600 hover:bg-green-500 px-4 py-2 rounded-lg text-sm font-medium transition-colors"
            >
              ⚙️ Settings
            </button>
            <button
              onClick={handleSignOut}
              className="bg-green-800 hover:bg-green-700 px-4 py-2 rounded-lg text-sm font-medium transition-colors"
            >
              Sign Out
            </button>
          </div>
        </div>
      </div>

      {/* Settings Modal */}
      {showSettings && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-xl p-6 max-w-md w-full mx-4 shadow-2xl">
            <h2 className="text-xl font-bold mb-4">Settings</h2>
            
            <div className="mb-4">
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Anthropic API Key
              </label>
              {apiKey ? (
                <div className="flex items-center gap-2">
                  <div className="flex-1 bg-gray-100 rounded-lg px-4 py-2 text-gray-600">
                    ••••••••••••{apiKey.slice(-8)}
                  </div>
                  <button
                    onClick={clearApiKey}
                    className="bg-red-100 hover:bg-red-200 text-red-700 px-4 py-2 rounded-lg font-medium"
                  >
                    Remove
                  </button>
                </div>
              ) : (
                <div className="flex gap-2">
                  <input
                    type="password"
                    value={apiKeyInput}
                    onChange={(e) => setApiKeyInput(e.target.value)}
                    placeholder="sk-ant-..."
                    className="flex-1 border border-gray-300 rounded-lg px-4 py-2 focus:outline-none focus:ring-2 focus:ring-green-500"
                  />
                  <button
                    onClick={saveApiKey}
                    disabled={!apiKeyInput.trim()}
                    className="bg-green-600 hover:bg-green-700 disabled:bg-gray-300 text-white px-4 py-2 rounded-lg font-medium"
                  >
                    Save
                  </button>
                </div>
              )}
              <p className="text-xs text-gray-500 mt-2">
                Stored in your Google Drive - synced across all devices.
              </p>
            </div>
            
            <div className="flex justify-end">
              <button
                onClick={() => setShowSettings(false)}
                className="bg-gray-200 hover:bg-gray-300 px-4 py-2 rounded-lg font-medium"
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}

      {/* API Key Warning */}
      {!apiKey && (
        <div className="bg-yellow-50 border-b border-yellow-200 px-6 py-3">
          <div className="max-w-6xl mx-auto flex items-center gap-2 text-yellow-800">
            <span>⚠️</span>
            <span>Please add your Anthropic API key in Settings to use the chat feature.</span>
            <button
              onClick={() => setShowSettings(true)}
              className="underline font-medium hover:text-yellow-900"
            >
              Open Settings
            </button>
          </div>
        </div>
      )}

      {/* Tabs */}
      <div className="bg-white border-b shadow-sm">
        <div className="max-w-6xl mx-auto flex">
          <button
            onClick={() => setActiveTab('chat')}
            className={`px-6 py-3 font-medium transition-colors ${
              activeTab === 'chat' 
                ? 'text-green-700 border-b-2 border-green-700' 
                : 'text-gray-500 hover:text-gray-700'
            }`}
          >
            💬 Chat
          </button>
          <button
            onClick={() => setActiveTab('manage')}
            className={`px-6 py-3 font-medium transition-colors ${
              activeTab === 'manage' 
                ? 'text-green-700 border-b-2 border-green-700' 
                : 'text-gray-500 hover:text-gray-700'
            }`}
          >
            📚 Manage Documents
          </button>
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 overflow-hidden">
        {activeTab === 'manage' ? (
          // ============ MANAGE TAB ============
          <div className="h-full overflow-y-auto p-6">
            <div className="max-w-4xl mx-auto space-y-6">
              
              {/* Add Category */}
              <div className="bg-white rounded-xl shadow-sm p-6">
                <h2 className="text-lg font-semibold mb-4">Create Category</h2>
                <div className="flex gap-3">
                  <input
                    type="text"
                    value={newCategoryName}
                    onChange={(e) => setNewCategoryName(e.target.value)}
                    onKeyDown={(e) => e.key === 'Enter' && addCategory()}
                    placeholder="e.g., NFHS, NCAA, MLB..."
                    className="flex-1 border border-gray-300 rounded-lg px-4 py-2 focus:outline-none focus:ring-2 focus:ring-green-500"
                  />
                  <button
                    onClick={addCategory}
                    disabled={!newCategoryName.trim()}
                    className="bg-green-600 hover:bg-green-700 disabled:bg-gray-300 text-white px-6 py-2 rounded-lg font-medium transition-colors"
                  >
                    Add Category
                  </button>
                </div>
              </div>

              {/* Categories List */}
              {categories.length === 0 ? (
                <div className="bg-white rounded-xl shadow-sm p-12 text-center text-gray-500">
                  <p className="text-4xl mb-4">📁</p>
                  <p className="text-lg">No categories yet</p>
                  <p className="mt-2">Create a category above to get started!</p>
                </div>
              ) : (
                categories.map(category => (
                  <div key={category.id} className="bg-white rounded-xl shadow-sm overflow-hidden">
                    <div className="bg-gray-50 px-6 py-4 flex items-center justify-between border-b">
                      <div>
                        <h3 className="font-semibold text-lg">{category.name}</h3>
                        <p className="text-sm text-gray-500">
                          {getCategoryDocs(category.id).length} document(s) • {getCategoryChunkCount(category.id)} chunks indexed
                        </p>
                      </div>
                      <div className="flex gap-2">
                        <input
                          type="file"
                          accept=".pdf"
                          onChange={(e) => {
                            if (e.target.files[0]) {
                              uploadDocument(e.target.files[0], category.id);
                              e.target.value = '';
                            }
                          }}
                          className="hidden"
                          id={`upload-${category.id}`}
                        />
                        <label
                          htmlFor={`upload-${category.id}`}
                          className="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-lg cursor-pointer font-medium transition-colors"
                        >
                          + Upload PDF
                        </label>
                        <button
                          onClick={() => deleteCategory(category.id)}
                          className="bg-red-100 hover:bg-red-200 text-red-700 px-4 py-2 rounded-lg font-medium transition-colors"
                        >
                          Delete
                        </button>
                      </div>
                    </div>
                    
                    <div className="p-4">
                      {getCategoryDocs(category.id).length === 0 ? (
                        <p className="text-gray-400 text-center py-4">No documents uploaded yet</p>
                      ) : (
                        <div className="space-y-2">
                          {getCategoryDocs(category.id).map(doc => (
                            <div key={doc.id} className="flex items-center justify-between bg-gray-50 rounded-lg px-4 py-3">
                              <div className="flex items-center gap-3">
                                <span className="text-2xl">📄</span>
                                <div>
                                  <p className="font-medium">{doc.name}</p>
                                  <p className="text-sm text-gray-500">
                                    {doc.pageCount} pages • {chunks.filter(c => c.documentId === doc.id).length} chunks
                                  </p>
                                </div>
                              </div>
                              <button
                                onClick={() => deleteDocument(doc.id)}
                                className="text-red-600 hover:text-red-800 text-sm font-medium"
                              >
                                Remove
                              </button>
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  </div>
                ))
              )}

              {/* Upload Progress */}
              {uploadProgress && (
                <div className="fixed bottom-6 right-6 bg-blue-600 text-white px-6 py-3 rounded-xl shadow-lg flex items-center gap-3">
                  <div className="animate-spin h-5 w-5 border-2 border-white border-t-transparent rounded-full"></div>
                  {uploadProgress}
                </div>
              )}
            </div>
          </div>
        ) : (
          // ============ CHAT TAB ============
          <div className="h-full flex flex-col">
            {/* Category Selector */}
            <div className="bg-white border-b px-6 py-3">
              <div className="max-w-4xl mx-auto flex items-center gap-4">
                <label className="font-medium text-gray-700">Rule Set:</label>
                <select
                  value={selectedCategory || ''}
                  onChange={(e) => {
                    setSelectedCategory(e.target.value || null);
                    setMessages([]);
                  }}
                  className="border border-gray-300 rounded-lg px-4 py-2 focus:outline-none focus:ring-2 focus:ring-green-500 min-w-48"
                >
                  <option value="">Select a category...</option>
                  {categories.map(cat => (
                    <option key={cat.id} value={cat.id}>
                      {cat.name} ({getCategoryDocs(cat.id).length} docs)
                    </option>
                  ))}
                </select>
                {selectedCategory && messages.length > 0 && (
                  <button
                    onClick={clearChat}
                    className="text-gray-500 hover:text-gray-700 text-sm font-medium"
                  >
                    Clear Chat
                  </button>
                )}
              </div>
            </div>

            {/* Messages */}
            <div className="flex-1 overflow-y-auto p-6">
              <div className="max-w-4xl mx-auto space-y-4">
                {!selectedCategory ? (
                  <div className="text-center py-12 text-gray-500">
                    <p className="text-4xl mb-4">👆</p>
                    <p className="text-lg">Select a rule set category above to start asking questions</p>
                    {categories.length === 0 && (
                      <p className="mt-2">Go to "Manage Documents" tab to create categories and upload PDFs first</p>
                    )}
                  </div>
                ) : messages.length === 0 ? (
                  <div className="text-center py-12 text-gray-500">
                    <p className="text-4xl mb-4">⚾</p>
                    <p className="text-lg">
                      Ready to answer questions about {categories.find(c => c.id === selectedCategory)?.name} rules!
                    </p>
                    <p className="mt-2">Ask me anything about the rules in your uploaded documents.</p>
                  </div>
                ) : (
                  messages.map((msg, idx) => (
                    <div key={idx} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                      <div className={`max-w-3xl rounded-2xl px-5 py-3 ${
                        msg.role === 'user'
                          ? 'bg-green-600 text-white rounded-br-md'
                          : 'bg-white shadow-md border border-gray-200 rounded-bl-md'
                      }`}>
                        <div className="whitespace-pre-wrap">{msg.content}</div>
                        {msg.sources && msg.sources.length > 0 && (
                          <div className="mt-3 pt-3 border-t border-gray-200 text-xs text-gray-500">
                            📚 Sources: {msg.sources.join(', ')}
                          </div>
                        )}
                      </div>
                    </div>
                  ))
                )}
                
                {isSending && (
                  <div className="flex justify-start">
                    <div className="bg-white shadow-md border border-gray-200 rounded-2xl rounded-bl-md px-5 py-3">
                      <div className="flex items-center gap-1 text-gray-500">
                        <span className="animate-pulse">●</span>
                        <span className="animate-pulse">●</span>
                        <span className="animate-pulse">●</span>
                        <span className="ml-2">Searching rule books...</span>
                      </div>
                    </div>
                  </div>
                )}
                <div ref={messagesEndRef} />
              </div>
            </div>

            {/* Input */}
            <div className="border-t bg-white p-4">
              <div className="max-w-4xl mx-auto flex gap-3">
                <input
                  type="text"
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyDown={(e) => e.key === 'Enter' && !e.shiftKey && sendMessage()}
                  placeholder={
                    !apiKey 
                      ? "Add your API key in Settings first..." 
                      : !selectedCategory 
                        ? "Select a category first..." 
                        : "Ask a rules question..."
                  }
                  disabled={!selectedCategory || isSending || !apiKey}
                  className="flex-1 border border-gray-300 rounded-xl px-4 py-3 focus:outline-none focus:ring-2 focus:ring-green-500 disabled:bg-gray-100"
                />
                <button
                  onClick={sendMessage}
                  disabled={!selectedCategory || isSending || !input.trim() || !apiKey}
                  className="bg-green-600 hover:bg-green-700 disabled:bg-gray-300 text-white px-6 py-3 rounded-xl font-medium transition-colors"
                >
                  Send
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
