import React, { useState, useEffect, useRef, useCallback } from 'react';
import * as pdfjsLib from 'pdfjs-dist';
import { searchChunks, BM25Index } from './search/index.js';
import { chunkText } from './chunking/index.js';
import Markdown from './components/Markdown.jsx';

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
const CHAT_HISTORY_FILE_NAME = 'chat-history.json';
const PINNED_ANSWERS_FILE_NAME = 'pinned-answers.json';

const STARTER_QUESTIONS = [
  "What is the infield fly rule?",
  "When can a runner steal a base?",
  "What's the difference between a balk and an illegal pitch?",
  "How does the designated hitter rule work?"
];

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
        } else {
          // No saved token - show sign-in screen
          onAuthChange(false);
        }
        resolve();
      };
      script.onerror = () => {
        console.error('Failed to load Google Identity Services');
        onAuthChange(false);
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
// chunkText imported from ./chunking/index.js
// searchChunks, BM25Index imported from ./search/index.js

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
  const [selectedModel, setSelectedModel] = useState(() => localStorage.getItem('selectedModel') || 'claude-sonnet-4-6');
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
  const [smartSearch, setSmartSearch] = useState(() => localStorage.getItem('smartSearch') !== 'false');
  const [isReindexing, setIsReindexing] = useState(false);
  const [expandedChunks, setExpandedChunks] = useState(new Set());
  const [chatHistory, setChatHistory] = useState({});
  const chatHistorySaveTimer = useRef(null);
  const previousCategoryRef = useRef(null);
  const messagesEndRef = useRef(null);
  const bm25IndexRef = useRef(new BM25Index());
  const textareaRef = useRef(null);
  const pdfCanvasRef = useRef(null);
  const [copiedIdx, setCopiedIdx] = useState(null);
  const [pdfViewerState, setPdfViewerState] = useState(null);
  const [comparisonMode, setComparisonMode] = useState(false);
  const [comparisonCategories, setComparisonCategories] = useState([null, null]);
  const [pinnedAnswers, setPinnedAnswers] = useState([]);
  const [showPinnedPanel, setShowPinnedPanel] = useState(false);

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

  // Rebuild BM25 index when chunks change
  useEffect(() => {
    if (chunks.length > 0) {
      bm25IndexRef.current.build(chunks);
      console.log(`BM25 index built with ${chunks.length} chunks`);
    }
  }, [chunks]);

  // Render PDF page to canvas when viewer state changes
  useEffect(() => {
    if (!pdfViewerState?.pdfDoc || !pdfCanvasRef.current) return;
    const renderPage = async () => {
      const page = await pdfViewerState.pdfDoc.getPage(pdfViewerState.currentPage);
      const canvas = pdfCanvasRef.current;
      const ctx = canvas.getContext('2d');
      const containerWidth = canvas.parentElement?.clientWidth || 600;
      const unscaledViewport = page.getViewport({ scale: 1 });
      const scale = containerWidth / unscaledViewport.width;
      const viewport = page.getViewport({ scale });
      canvas.height = viewport.height;
      canvas.width = viewport.width;
      await page.render({ canvasContext: ctx, viewport }).promise;
    };
    renderPage();
  }, [pdfViewerState?.pdfDoc, pdfViewerState?.currentPage]);

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

      const historyData = await driveService.readJsonFile(CHAT_HISTORY_FILE_NAME);
      if (historyData) {
        setChatHistory(historyData);
      }

      const pinnedData = await driveService.readJsonFile(PINNED_ANSWERS_FILE_NAME);
      if (pinnedData) {
        setPinnedAnswers(pinnedData);
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

  const saveChatHistoryToDrive = useCallback(async (history) => {
    try {
      await driveService.writeJsonFile(CHAT_HISTORY_FILE_NAME, history);
    } catch (error) {
      console.error('Error saving chat history:', error);
    }
  }, []);

  const savePinnedAnswersToDrive = useCallback(async (pins) => {
    try {
      await driveService.writeJsonFile(PINNED_ANSWERS_FILE_NAME, pins);
    } catch (error) {
      console.error('Error saving pinned answers:', error);
    }
  }, []);

  const debouncedSaveChatHistory = useCallback((history) => {
    if (chatHistorySaveTimer.current) {
      clearTimeout(chatHistorySaveTimer.current);
    }
    chatHistorySaveTimer.current = setTimeout(() => {
      saveChatHistoryToDrive(history);
    }, 2000);
  }, [saveChatHistoryToDrive]);

  // Auto-save messages to chat history when messages change
  useEffect(() => {
    if (selectedCategory && messages.length > 0 && !isSending && !comparisonMode) {
      const lightweight = messages.map(m => ({ role: m.role, content: m.content }));
      setChatHistory(prev => {
        const updated = {
          ...prev,
          [selectedCategory]: { messages: lightweight, updatedAt: new Date().toISOString() }
        };
        debouncedSaveChatHistory(updated);
        return updated;
      });
    }
  }, [messages, isSending, selectedCategory, debouncedSaveChatHistory, comparisonMode]);

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
    setChatHistory({});
    setPinnedAnswers([]);
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
      const textChunks = chunkText(text);

      const newChunks = textChunks.map((chunk, i) => ({
        id: `${document.id}_${i}`,
        documentId: document.id,
        categoryId,
        text: chunk.text,
        ruleRef: chunk.ruleRef,
        sectionType: chunk.sectionType,
        pageNumber: chunk.pageNumber,
        index: i
      }));

      console.log(`Created ${newChunks.length} rule-aware chunks`,
        newChunks.filter(c => c.ruleRef).length, 'with rule refs',
        newChunks.filter(c => c.sectionType !== 'rule').length, 'with special section types');
      
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

  const sendMessage = async (overrideText) => {
    const text = (typeof overrideText === 'string' ? overrideText : input).trim();
    if (!text || isSending || !apiKey) return;

    // In comparison mode, need both categories selected
    if (comparisonMode) {
      if (!comparisonCategories[0] || !comparisonCategories[1]) return;
    } else {
      if (!selectedCategory) return;
    }

    const userMessage = { role: 'user', content: text };
    const newMessages = [...messages, userMessage];
    setMessages(newMessages);
    setInput('');
    if (textareaRef.current) {
      textareaRef.current.style.height = '48px';
    }
    setIsSending(true);

    try {
      let categoryChunks;
      let isAllCategories;
      let categoryName;

      if (comparisonMode) {
        categoryChunks = chunks.filter(c => comparisonCategories.includes(c.categoryId));
        isAllCategories = false;
        const cat1Name = categories.find(c => c.id === comparisonCategories[0])?.name || 'Unknown';
        const cat2Name = categories.find(c => c.id === comparisonCategories[1])?.name || 'Unknown';
        categoryName = `${cat1Name} vs ${cat2Name}`;
      } else {
        isAllCategories = selectedCategory === '__all__';
        categoryChunks = isAllCategories
          ? chunks
          : chunks.filter(c => c.categoryId === selectedCategory);
        categoryName = isAllCategories
          ? `all categories (${categories.map(c => c.name).join(', ')})`
          : (categories.find(c => c.id === selectedCategory)?.name || 'Unknown');
      }

      if (categoryChunks.length === 0) {
        setMessages([...newMessages, {
          role: 'assistant',
          content: "I don't have any documents loaded for this category yet. Please upload some PDFs first in the Manage tab."
        }]);
        setIsSending(false);
        return;
      }

      // Build a category-specific BM25 index
      const categoryIndex = new BM25Index();
      categoryIndex.build(categoryChunks);

      const relevantChunks = await searchChunks({
        bm25Index: categoryIndex,
        chunks: categoryChunks,
        query: text,
        topK: 8,
        smartSearch,
        apiKey,
      });

      const context = relevantChunks.map((chunk) => {
        const doc = documents.find(d => d.id === chunk.documentId);
        const catName = (isAllCategories || comparisonMode)
          ? categories.find(c => c.id === chunk.categoryId)?.name
          : null;
        const sourceLabel = catName ? `${catName} - ${doc?.name || 'Unknown'}` : (doc?.name || 'Unknown');
        return `[Source: ${sourceLabel}]\n${chunk.text}`;
      }).join('\n\n---\n\n');

      let systemPrompt;
      if (comparisonMode) {
        const cat1Name = categories.find(c => c.id === comparisonCategories[0])?.name || 'Unknown';
        const cat2Name = categories.find(c => c.id === comparisonCategories[1])?.name || 'Unknown';
        systemPrompt = `You are an expert baseball rules advisor comparing rules between ${cat1Name} and ${cat2Name}.

You have access to the following reference material from both rule sets:

${context}

Guidelines:
1. Compare and contrast the rules between ${cat1Name} and ${cat2Name}
2. Clearly indicate which rule set each point comes from
3. Highlight key differences and similarities
4. Cite specific rules when possible (e.g., "Rule 8-4-2b")
5. If one rule set covers something the other doesn't, note that
6. Format responses with clear headers for each rule set
7. Be conversational but precise`;
      } else {
        systemPrompt = `You are an expert baseball rules advisor specializing in ${categoryName} rules.

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
      }

      // Build rich source details
      const sourceDetails = [];
      const seenSourceKeys = new Set();
      for (const c of relevantChunks) {
        const doc = documents.find(d => d.id === c.documentId);
        const docName = doc?.name || 'Unknown';
        const catName = (isAllCategories || comparisonMode) ? categories.find(cat => cat.id === c.categoryId)?.name : null;
        const dedupeKey = `${c.ruleRef || ''}_${c.pageNumber || ''}_${docName}`;
        if (!seenSourceKeys.has(dedupeKey)) {
          seenSourceKeys.add(dedupeKey);
          sourceDetails.push({
            docName,
            categoryName: catName,
            ruleRef: c.ruleRef,
            pageNumber: c.pageNumber,
            score: c.score,
            documentId: c.documentId,
            driveFileId: doc?.driveFileId
          });
        }
      }

      // Build search results for transparency panel (Feature 4)
      const searchResults = relevantChunks.map(c => ({
        text: c.text.slice(0, 300),
        score: typeof c.score === 'number' ? c.score.toFixed(1) : '0',
        ruleRef: c.ruleRef,
        sectionType: c.sectionType,
        pageNumber: c.pageNumber
      }));

      // Add placeholder assistant message for streaming
      const assistantMessage = {
        role: 'assistant',
        content: '',
        sources: sourceDetails,
        searchResults
      };
      setMessages([...newMessages, assistantMessage]);

      const maxRetries = 3;
      let lastError = null;
      let streamStarted = false;

      for (let attempt = 0; attempt < maxRetries; attempt++) {
        if (attempt > 0) {
          const delay = Math.min(1000 * Math.pow(2, attempt), 8000);
          await new Promise(r => setTimeout(r, delay));
        }

        const response = await fetch('https://api.anthropic.com/v1/messages', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'x-api-key': apiKey,
            'anthropic-version': '2023-06-01',
            'anthropic-dangerous-direct-browser-access': 'true'
          },
          body: JSON.stringify({
            model: selectedModel,
            max_tokens: 2000,
            stream: true,
            system: systemPrompt,
            messages: newMessages.slice(-10).map(m => ({ role: m.role, content: m.content }))
          })
        });

        if (!response.ok) {
          if (response.status === 529 || response.status === 429) {
            lastError = { message: `API error ${response.status}` };
            continue;
          }
          const errorData = await response.json().catch(() => ({}));
          throw new Error(errorData.error?.message || `API error ${response.status}`);
        }

        // Stream the response
        streamStarted = true;
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let accumulated = '';
        let buffer = '';

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split('\n');
          buffer = lines.pop() || '';

          for (const line of lines) {
            if (!line.startsWith('data: ')) continue;
            const jsonStr = line.slice(6).trim();
            if (jsonStr === '[DONE]') continue;

            try {
              const event = JSON.parse(jsonStr);
              if (event.type === 'content_block_delta' && event.delta?.text) {
                accumulated += event.delta.text;
                const textSoFar = accumulated;
                setMessages(prev => {
                  const updated = [...prev];
                  const last = updated[updated.length - 1];
                  if (last && last.role === 'assistant') {
                    updated[updated.length - 1] = { ...last, content: textSoFar };
                  }
                  return updated;
                });
              }
            } catch {
              // Skip malformed JSON
            }
          }
        }

        // Finalize - ensure final content is set
        if (accumulated) {
          const finalText = accumulated;
          const finalSources = sourceDetails;
          const finalSearchResults = searchResults;
          setMessages(prev => {
            const updated = [...prev];
            const last = updated[updated.length - 1];
            if (last && last.role === 'assistant') {
              updated[updated.length - 1] = {
                ...last,
                content: finalText,
                sources: finalSources,
                searchResults: finalSearchResults
              };
            }
            return updated;
          });
          return;
        }
      }

      // All retries exhausted
      if (!streamStarted) {
        throw new Error(lastError?.message || 'The API is currently overloaded. Please try again in a moment.');
      }
    } catch (error) {
      const msg = error.message?.toLowerCase() || '';
      let errorContent;
      if (msg.includes('overloaded')) {
        errorContent = 'The AI service is temporarily overloaded. Please wait a moment and try again.';
      } else if (msg.includes('invalid') || msg.includes('authentication') || msg.includes('unauthorized')) {
        errorContent = 'API key error. Please check your API key in Settings.';
      } else if (msg.includes('rate')) {
        errorContent = 'Rate limit reached. Please wait a moment and try again.';
      } else {
        errorContent = `Error: ${error.message}`;
      }
      setMessages(prev => {
        const updated = [...prev];
        const last = updated[updated.length - 1];
        if (last && last.role === 'assistant') {
          updated[updated.length - 1] = { ...last, content: errorContent };
        } else {
          updated.push({ role: 'assistant', content: errorContent });
        }
        return updated;
      });
    } finally {
      setIsSending(false);
    }
  };

  const reindexDocuments = async () => {
    if (!confirm('Re-index all documents with improved rule-aware chunking? This will re-download and re-process all PDFs.')) return;

    setIsReindexing(true);
    setUploadProgress('Re-indexing documents...');

    try {
      let allNewChunks = [];

      for (let i = 0; i < documents.length; i++) {
        const doc = documents[i];
        setUploadProgress(`Re-indexing ${doc.name} (${i + 1}/${documents.length})...`);

        if (!doc.driveFileId) continue;

        try {
          const arrayBuffer = await driveService.downloadPdf(doc.driveFileId);
          const { text } = await extractTextFromPDF(arrayBuffer);
          const textChunks = chunkText(text);

          const docChunks = textChunks.map((chunk, idx) => ({
            id: `${doc.id}_${idx}`,
            documentId: doc.id,
            categoryId: doc.categoryId,
            text: chunk.text,
            ruleRef: chunk.ruleRef,
            sectionType: chunk.sectionType,
            pageNumber: chunk.pageNumber,
            index: idx
          }));

          allNewChunks = [...allNewChunks, ...docChunks];
        } catch (err) {
          console.error(`Error re-indexing ${doc.name}:`, err);
          // Keep old chunks for this doc
          const oldDocChunks = chunks.filter(c => c.documentId === doc.id);
          allNewChunks = [...allNewChunks, ...oldDocChunks];
        }
      }

      setChunks(allNewChunks);
      setUploadProgress('Saving re-indexed chunks...');
      await saveChunksToDrive(allNewChunks);

      console.log(`Re-indexed: ${allNewChunks.length} total chunks`,
        allNewChunks.filter(c => c.ruleRef).length, 'with rule refs');

      setUploadProgress(null);
    } catch (error) {
      console.error('Re-index error:', error);
      setUploadProgress(`Re-index error: ${error.message}`);
      setTimeout(() => setUploadProgress(null), 3000);
    } finally {
      setIsReindexing(false);
    }
  };

  const clearChat = () => {
    setMessages([]);
    if (selectedCategory) {
      setChatHistory(prev => {
        const updated = { ...prev };
        delete updated[selectedCategory];
        debouncedSaveChatHistory(updated);
        return updated;
      });
    }
  };

  const exportChat = () => {
    const catName = comparisonMode
      ? `${categories.find(c => c.id === comparisonCategories[0])?.name || 'Unknown'} vs ${categories.find(c => c.id === comparisonCategories[1])?.name || 'Unknown'}`
      : selectedCategory === '__all__'
        ? 'All Categories'
        : (categories.find(c => c.id === selectedCategory)?.name || 'Unknown');
    const header = `Baseball Rules Assistant - ${catName}\nExported: ${new Date().toLocaleString()}\n${'='.repeat(50)}\n\n`;
    const body = messages.map(m => {
      let text = `[${m.role === 'user' ? 'You' : 'Assistant'}]\n${m.content}`;
      if (m.sources && m.sources.length > 0) {
        const srcLabels = m.sources.map(src => {
          const prefix = src.categoryName ? `[${src.categoryName}] ` : '';
          if (src.ruleRef && src.pageNumber) return `${prefix}${src.ruleRef} (p.${src.pageNumber})`;
          if (src.pageNumber) return `${prefix}${src.docName} p.${src.pageNumber}`;
          if (src.ruleRef) return `${prefix}${src.ruleRef}`;
          return `${prefix}${src.docName}`;
        });
        text += `\nSources: ${srcLabels.join(', ')}`;
      }
      return text;
    }).join('\n\n---\n\n');
    const blob = new Blob([header + body], { type: 'text/plain' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `rules-chat-${catName.replace(/\s+/g, '-').toLowerCase()}-${new Date().toISOString().slice(0, 10)}.txt`;
    a.click();
    URL.revokeObjectURL(a.href);
  };

  const openPdfPage = async (documentId, pageNumber) => {
    const doc = documents.find(d => d.id === documentId);
    if (!doc?.driveFileId) return;
    setPdfViewerState({ docName: doc.name, pageNumber, loading: true, pdfDoc: null, currentPage: pageNumber, totalPages: 0 });
    try {
      const arrayBuffer = await driveService.downloadPdf(doc.driveFileId);
      const pdfDoc = await pdfjsLib.getDocument(new Uint8Array(arrayBuffer)).promise;
      setPdfViewerState(prev => prev ? { ...prev, pdfDoc, totalPages: pdfDoc.numPages, loading: false } : null);
    } catch (err) {
      console.error('Error opening PDF:', err);
      setPdfViewerState(null);
    }
  };

  const pinAnswer = (messageIdx) => {
    const msg = messages[messageIdx];
    if (!msg || msg.role !== 'assistant') return;
    const userMsg = messages[messageIdx - 1];
    const catName = comparisonMode
      ? `${categories.find(c => c.id === comparisonCategories[0])?.name} vs ${categories.find(c => c.id === comparisonCategories[1])?.name}`
      : selectedCategory === '__all__'
        ? 'All Categories'
        : (categories.find(c => c.id === selectedCategory)?.name || 'Unknown');
    const pin = {
      id: Date.now().toString(),
      question: userMsg?.content || '',
      answer: msg.content,
      sources: msg.sources || [],
      categoryName: catName,
      categoryId: selectedCategory,
      timestamp: new Date().toISOString()
    };
    const updated = [pin, ...pinnedAnswers];
    setPinnedAnswers(updated);
    savePinnedAnswersToDrive(updated);
  };

  const unpinAnswer = (pinId) => {
    const updated = pinnedAnswers.filter(p => p.id !== pinId);
    setPinnedAnswers(updated);
    savePinnedAnswersToDrive(updated);
  };

  const isAnswerPinned = (messageIdx) => {
    const msg = messages[messageIdx];
    if (!msg || msg.role !== 'assistant') return false;
    return pinnedAnswers.some(p => p.answer === msg.content && (p.categoryId === selectedCategory || comparisonMode));
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

            <div className="mb-4">
              <label className="block text-sm font-medium text-gray-700 mb-2">
                AI Model
              </label>
              <select
                value={selectedModel}
                onChange={(e) => {
                  setSelectedModel(e.target.value);
                  localStorage.setItem('selectedModel', e.target.value);
                }}
                className="w-full border border-gray-300 rounded-lg px-4 py-2 focus:outline-none focus:ring-2 focus:ring-green-500 bg-white"
              >
                <option value="claude-sonnet-4-6">Claude Sonnet 4.6 (Recommended)</option>
                <option value="claude-haiku-4-5-20251001">Claude Haiku 4.5 (Faster, cheaper)</option>
                <option value="claude-sonnet-4-20250514">Claude Sonnet 4 (Legacy)</option>
              </select>
              <p className="text-xs text-gray-500 mt-2">
                If you're getting overload errors, try switching models.
              </p>
            </div>

            <div className="mb-4">
              <label className="flex items-center justify-between">
                <div>
                  <span className="text-sm font-medium text-gray-700">Smart Search</span>
                  <p className="text-xs text-gray-500 mt-1">
                    Uses AI to expand your questions with better search terms. Adds ~1s latency per question.
                  </p>
                </div>
                <button
                  onClick={() => {
                    const newVal = !smartSearch;
                    setSmartSearch(newVal);
                    localStorage.setItem('smartSearch', String(newVal));
                  }}
                  className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ml-4 flex-shrink-0 ${
                    smartSearch ? 'bg-green-600' : 'bg-gray-300'
                  }`}
                >
                  <span
                    className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                      smartSearch ? 'translate-x-6' : 'translate-x-1'
                    }`}
                  />
                </button>
              </label>
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

              {/* Re-index Button */}
              {documents.length > 0 && (
                <div className="bg-white rounded-xl shadow-sm p-6">
                  <div className="flex items-center justify-between">
                    <div>
                      <h2 className="text-lg font-semibold">Re-index Documents</h2>
                      <p className="text-sm text-gray-500 mt-1">
                        Re-process all PDFs with improved rule-aware chunking for better search quality.
                      </p>
                    </div>
                    <button
                      onClick={reindexDocuments}
                      disabled={isReindexing}
                      className="bg-amber-600 hover:bg-amber-700 disabled:bg-gray-300 text-white px-6 py-2 rounded-lg font-medium transition-colors flex-shrink-0"
                    >
                      {isReindexing ? 'Re-indexing...' : 'Re-index All'}
                    </button>
                  </div>
                </div>
              )}

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
              <div className="max-w-4xl mx-auto flex flex-wrap items-center gap-4">
                {!comparisonMode ? (
                  <>
                    <label className="font-medium text-gray-700">Rule Set:</label>
                    <select
                      value={selectedCategory || ''}
                      onChange={(e) => {
                        const newCat = e.target.value || null;
                        if (selectedCategory && messages.length > 0) {
                          const lightweight = messages.map(m => ({ role: m.role, content: m.content }));
                          setChatHistory(prev => {
                            const updated = {
                              ...prev,
                              [selectedCategory]: { messages: lightweight, updatedAt: new Date().toISOString() }
                            };
                            debouncedSaveChatHistory(updated);
                            return updated;
                          });
                        }
                        setSelectedCategory(newCat);
                        setExpandedChunks(new Set());
                        if (newCat && chatHistory[newCat]?.messages) {
                          setMessages(chatHistory[newCat].messages);
                        } else {
                          setMessages([]);
                        }
                      }}
                      className="border border-gray-300 rounded-lg px-4 py-2 focus:outline-none focus:ring-2 focus:ring-green-500 min-w-48"
                    >
                      <option value="">Select a category...</option>
                      {categories.length >= 2 && (
                        <option value="__all__">All Categories</option>
                      )}
                      {categories.map(cat => (
                        <option key={cat.id} value={cat.id}>
                          {cat.name} ({getCategoryDocs(cat.id).length} docs)
                        </option>
                      ))}
                    </select>
                  </>
                ) : (
                  <>
                    <label className="font-medium text-gray-700">Compare:</label>
                    <select
                      value={comparisonCategories[0] || ''}
                      onChange={(e) => {
                        setComparisonCategories(prev => [e.target.value || null, prev[1]]);
                        setMessages([]);
                      }}
                      className="border border-gray-300 rounded-lg px-4 py-2 focus:outline-none focus:ring-2 focus:ring-green-500"
                    >
                      <option value="">Select...</option>
                      {categories.map(cat => (
                        <option key={cat.id} value={cat.id} disabled={cat.id === comparisonCategories[1]}>
                          {cat.name}
                        </option>
                      ))}
                    </select>
                    <span className="font-bold text-gray-500">vs</span>
                    <select
                      value={comparisonCategories[1] || ''}
                      onChange={(e) => {
                        setComparisonCategories(prev => [prev[0], e.target.value || null]);
                        setMessages([]);
                      }}
                      className="border border-gray-300 rounded-lg px-4 py-2 focus:outline-none focus:ring-2 focus:ring-green-500"
                    >
                      <option value="">Select...</option>
                      {categories.map(cat => (
                        <option key={cat.id} value={cat.id} disabled={cat.id === comparisonCategories[0]}>
                          {cat.name}
                        </option>
                      ))}
                    </select>
                  </>
                )}
                {categories.length >= 2 && (
                  <button
                    onClick={() => {
                      setComparisonMode(!comparisonMode);
                      setMessages([]);
                      if (!comparisonMode) {
                        setSelectedCategory(null);
                      } else {
                        setComparisonCategories([null, null]);
                      }
                    }}
                    className={`px-3 py-2 rounded-lg text-sm font-medium transition-colors ${
                      comparisonMode
                        ? 'bg-purple-100 text-purple-700 border border-purple-300'
                        : 'bg-gray-100 text-gray-600 hover:bg-gray-200 border border-gray-300'
                    }`}
                  >
                    {comparisonMode ? 'Exit Compare' : 'Compare'}
                  </button>
                )}
                <div className="flex items-center gap-2">
                  {pinnedAnswers.length > 0 && (
                    <button
                      onClick={() => setShowPinnedPanel(!showPinnedPanel)}
                      className={`px-3 py-2 rounded-lg text-sm font-medium transition-colors ${
                        showPinnedPanel
                          ? 'bg-amber-100 text-amber-700 border border-amber-300'
                          : 'bg-gray-100 text-gray-600 hover:bg-gray-200 border border-gray-300'
                      }`}
                    >
                      Pins ({pinnedAnswers.length})
                    </button>
                  )}
                  {((selectedCategory && messages.length > 0) || (comparisonMode && messages.length > 0)) && (
                    <>
                      <button
                        onClick={exportChat}
                        className="text-gray-500 hover:text-gray-700 text-sm font-medium"
                      >
                        Export
                      </button>
                      <button
                        onClick={clearChat}
                        className="text-gray-500 hover:text-gray-700 text-sm font-medium"
                      >
                        Clear Chat
                      </button>
                    </>
                  )}
                </div>
              </div>
            </div>

            {/* Pinned Answers Panel */}
            {showPinnedPanel && pinnedAnswers.length > 0 && (
              <div className="bg-amber-50 border-b border-amber-200 px-6 py-3 max-h-64 overflow-y-auto">
                <div className="max-w-4xl mx-auto space-y-2">
                  {pinnedAnswers.map(pin => (
                    <div key={pin.id} className="bg-white rounded-lg border border-amber-200 p-3">
                      <div className="flex items-start justify-between gap-2">
                        <div className="flex-1 min-w-0">
                          <div className="flex items-center gap-2 mb-1">
                            <span className="text-xs bg-amber-100 text-amber-700 px-2 py-0.5 rounded-full">{pin.categoryName}</span>
                            <span className="text-xs text-gray-400">{new Date(pin.timestamp).toLocaleDateString()}</span>
                          </div>
                          <p className="text-sm font-medium text-gray-800 truncate">{pin.question}</p>
                          <div className="text-xs text-gray-600 mt-1 line-clamp-2">
                            <Markdown content={pin.answer.slice(0, 200) + (pin.answer.length > 200 ? '...' : '')} />
                          </div>
                        </div>
                        <button
                          onClick={() => unpinAnswer(pin.id)}
                          className="text-amber-500 hover:text-amber-700 text-sm flex-shrink-0"
                          title="Unpin"
                        >
                          Unpin
                        </button>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Messages */}
            <div className="flex-1 overflow-y-auto p-6">
              <div className="max-w-4xl mx-auto space-y-4">
                {(comparisonMode ? (!comparisonCategories[0] || !comparisonCategories[1]) : !selectedCategory) ? (
                  <div className="text-center py-12 text-gray-500">
                    <p className="text-4xl mb-4">👆</p>
                    <p className="text-lg">{comparisonMode ? 'Select two categories above to compare rules' : 'Select a rule set category above to start asking questions'}</p>
                    {categories.length === 0 && !comparisonMode && (
                      <p className="mt-2">Go to "Manage Documents" tab to create categories and upload PDFs first</p>
                    )}
                  </div>
                ) : messages.length === 0 ? (
                  <div className="text-center py-12 text-gray-500">
                    <p className="text-4xl mb-4">⚾</p>
                    <p className="text-lg">
                      Ready to answer questions about {
                        comparisonMode
                          ? `${categories.find(c => c.id === comparisonCategories[0])?.name || '...'} vs ${categories.find(c => c.id === comparisonCategories[1])?.name || '...'}`
                          : selectedCategory === '__all__' ? 'all categories' : categories.find(c => c.id === selectedCategory)?.name
                      } rules!
                    </p>
                    <p className="mt-2 mb-4">Try one of these questions, or ask your own:</p>
                    <div className="flex flex-wrap justify-center gap-2">
                      {STARTER_QUESTIONS.map((q, qi) => (
                        <button
                          key={qi}
                          onClick={() => sendMessage(q)}
                          disabled={isSending || !apiKey}
                          className="bg-white border border-green-300 text-green-800 rounded-full px-4 py-2 text-sm hover:bg-green-50 transition-colors disabled:opacity-50"
                        >
                          {q}
                        </button>
                      ))}
                    </div>
                  </div>
                ) : (
                  messages.map((msg, idx) => (
                    <div key={idx} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                      <div className={`max-w-3xl rounded-2xl px-5 py-3 ${
                        msg.role === 'user'
                          ? 'bg-green-600 text-white rounded-br-md'
                          : 'bg-white shadow-md border border-gray-200 rounded-bl-md'
                      }`}>
                        {msg.role === 'user' ? (
                          <div className="whitespace-pre-wrap">{msg.content}</div>
                        ) : (
                          <>
                            <Markdown content={msg.content} />
                            {msg.content && (
                              <div className="flex items-center gap-2 mt-2 -mb-1">
                                <button
                                  onClick={() => {
                                    navigator.clipboard.writeText(msg.content);
                                    setCopiedIdx(idx);
                                    setTimeout(() => setCopiedIdx(null), 2000);
                                  }}
                                  className="text-xs text-gray-400 hover:text-gray-600 transition-colors"
                                >
                                  {copiedIdx === idx ? 'Copied!' : 'Copy'}
                                </button>
                                <button
                                  onClick={() => isAnswerPinned(idx) ? unpinAnswer(pinnedAnswers.find(p => p.answer === msg.content)?.id) : pinAnswer(idx)}
                                  className={`text-xs transition-colors ${
                                    isAnswerPinned(idx) ? 'text-amber-500 hover:text-amber-700' : 'text-gray-400 hover:text-gray-600'
                                  }`}
                                >
                                  {isAnswerPinned(idx) ? 'Pinned' : 'Pin'}
                                </button>
                              </div>
                            )}
                          </>
                        )}
                        {/* Source Citations (Feature 3) */}
                        {msg.sources && msg.sources.length > 0 && (
                          <div className="mt-3 pt-3 border-t border-gray-200">
                            <div className="flex flex-wrap gap-1.5">
                              {(Array.isArray(msg.sources) && typeof msg.sources[0] === 'object'
                                ? msg.sources
                                : msg.sources.map(s => ({ docName: s }))
                              ).map((src, si) => {
                                let label;
                                const prefix = src.categoryName ? `[${src.categoryName}] ` : '';
                                if (src.ruleRef && src.pageNumber) {
                                  label = `${prefix}${src.ruleRef} (p.${src.pageNumber})`;
                                } else if (src.pageNumber) {
                                  label = `${prefix}${src.docName} p.${src.pageNumber}`;
                                } else if (src.ruleRef) {
                                  label = `${prefix}${src.ruleRef}`;
                                } else {
                                  label = `${prefix}${src.docName}`;
                                }
                                const clickable = src.documentId && src.pageNumber && src.driveFileId;
                                return clickable ? (
                                  <button
                                    key={si}
                                    onClick={() => openPdfPage(src.documentId, src.pageNumber)}
                                    className="inline-block bg-green-50 text-green-800 text-xs px-2 py-1 rounded-full border border-green-200 hover:bg-green-100 transition-colors cursor-pointer"
                                  >
                                    📚 {label}
                                  </button>
                                ) : (
                                  <span
                                    key={si}
                                    className="inline-block bg-green-50 text-green-800 text-xs px-2 py-1 rounded-full border border-green-200"
                                  >
                                    📚 {label}
                                  </span>
                                );
                              })}
                            </div>
                          </div>
                        )}
                        {/* Search Transparency Panel (Feature 4) */}
                        {msg.searchResults && msg.searchResults.length > 0 && (
                          <div className="mt-2">
                            <button
                              onClick={() => setExpandedChunks(prev => {
                                const next = new Set(prev);
                                if (next.has(idx)) next.delete(idx);
                                else next.add(idx);
                                return next;
                              })}
                              className="text-xs text-gray-400 hover:text-gray-600 transition-colors"
                            >
                              {expandedChunks.has(idx) ? '▼ Hide' : '▶ Show'} search results ({msg.searchResults.length})
                            </button>
                            {expandedChunks.has(idx) && (
                              <div className="mt-2 space-y-2">
                                {[...msg.searchResults]
                                  .sort((a, b) => parseFloat(b.score) - parseFloat(a.score))
                                  .map((sr, sri) => (
                                    <div key={sri} className="bg-gray-50 border border-gray-200 rounded-lg p-3 text-xs">
                                      <div className="flex items-center gap-2 mb-1">
                                        <span className="bg-blue-100 text-blue-700 px-1.5 py-0.5 rounded font-mono font-medium">
                                          {sr.score}
                                        </span>
                                        {sr.ruleRef && (
                                          <span className="bg-green-100 text-green-700 px-1.5 py-0.5 rounded">
                                            {sr.ruleRef}
                                          </span>
                                        )}
                                        {sr.sectionType && sr.sectionType !== 'rule' && (
                                          <span className="bg-purple-100 text-purple-700 px-1.5 py-0.5 rounded">
                                            {sr.sectionType}
                                          </span>
                                        )}
                                        {sr.pageNumber && (
                                          <span className="text-gray-500">p.{sr.pageNumber}</span>
                                        )}
                                      </div>
                                      <p className="text-gray-600 leading-snug">
                                        {sr.text.length > 200 ? sr.text.slice(0, 200) + '...' : sr.text}
                                      </p>
                                    </div>
                                  ))}
                              </div>
                            )}
                          </div>
                        )}
                      </div>
                    </div>
                  ))
                )}
                
                {isSending && (messages.length === 0 || messages[messages.length - 1]?.role !== 'assistant' || !messages[messages.length - 1]?.content) && (
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
                <textarea
                  ref={textareaRef}
                  rows={1}
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onInput={(e) => {
                    e.target.style.height = '48px';
                    e.target.style.height = Math.min(e.target.scrollHeight, 150) + 'px';
                  }}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' && !e.shiftKey) {
                      e.preventDefault();
                      sendMessage();
                    }
                  }}
                  placeholder={
                    !apiKey
                      ? "Add your API key in Settings first..."
                      : (!selectedCategory && !comparisonMode)
                        ? "Select a category first..."
                        : "Ask a rules question... (Shift+Enter for new line)"
                  }
                  disabled={comparisonMode ? (!comparisonCategories[0] || !comparisonCategories[1] || isSending || !apiKey) : (!selectedCategory || isSending || !apiKey)}
                  className="flex-1 border border-gray-300 rounded-xl px-4 py-3 focus:outline-none focus:ring-2 focus:ring-green-500 disabled:bg-gray-100 resize-none overflow-hidden"
                  style={{ minHeight: '48px', maxHeight: '150px' }}
                />
                <button
                  onClick={() => sendMessage()}
                  disabled={comparisonMode ? (!comparisonCategories[0] || !comparisonCategories[1] || isSending || !input.trim() || !apiKey) : (!selectedCategory || isSending || !input.trim() || !apiKey)}
                  className="bg-green-600 hover:bg-green-700 disabled:bg-gray-300 text-white px-6 py-3 rounded-xl font-medium transition-colors"
                >
                  Send
                </button>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* PDF Viewer Modal */}
      {pdfViewerState && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-xl shadow-2xl max-w-3xl w-full mx-4 max-h-[90vh] flex flex-col">
            <div className="flex items-center justify-between px-6 py-4 border-b">
              <div>
                <h3 className="font-semibold">{pdfViewerState.docName}</h3>
                {pdfViewerState.totalPages > 0 && (
                  <p className="text-sm text-gray-500">Page {pdfViewerState.currentPage} of {pdfViewerState.totalPages}</p>
                )}
              </div>
              <div className="flex items-center gap-2">
                {pdfViewerState.totalPages > 0 && (
                  <>
                    <button
                      onClick={() => setPdfViewerState(prev => prev && prev.currentPage > 1 ? { ...prev, currentPage: prev.currentPage - 1 } : prev)}
                      disabled={!pdfViewerState.pdfDoc || pdfViewerState.currentPage <= 1}
                      className="px-3 py-1 rounded bg-gray-100 hover:bg-gray-200 disabled:opacity-50 text-sm"
                    >
                      Prev
                    </button>
                    <button
                      onClick={() => setPdfViewerState(prev => prev && prev.currentPage < prev.totalPages ? { ...prev, currentPage: prev.currentPage + 1 } : prev)}
                      disabled={!pdfViewerState.pdfDoc || pdfViewerState.currentPage >= pdfViewerState.totalPages}
                      className="px-3 py-1 rounded bg-gray-100 hover:bg-gray-200 disabled:opacity-50 text-sm"
                    >
                      Next
                    </button>
                  </>
                )}
                <button
                  onClick={() => setPdfViewerState(null)}
                  className="ml-2 text-gray-500 hover:text-gray-700 text-xl font-bold"
                >
                  ×
                </button>
              </div>
            </div>
            <div className="flex-1 overflow-auto p-4">
              {pdfViewerState.loading ? (
                <div className="text-center py-12">
                  <div className="animate-spin text-4xl mb-4">⚾</div>
                  <p className="text-gray-500">Loading PDF...</p>
                </div>
              ) : (
                <canvas ref={pdfCanvasRef} className="mx-auto" />
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
