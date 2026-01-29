import React, { useState, useEffect, useRef } from 'react';
import * as pdfjsLib from 'pdfjs-dist';

// Set up PDF.js worker for v5+
pdfjsLib.GlobalWorkerOptions.workerSrc = new URL(
  'pdfjs-dist/build/pdf.worker.min.mjs',
  import.meta.url
).toString();

// ============ IndexedDB Helper ============
const DB_NAME = 'BaseballRulesRAG';
const DB_VERSION = 1;

const openDB = () => {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, DB_VERSION);
    request.onerror = () => reject(request.error);
    request.onsuccess = () => resolve(request.result);
    request.onupgradeneeded = (event) => {
      const db = event.target.result;
      if (!db.objectStoreNames.contains('categories')) {
        db.createObjectStore('categories', { keyPath: 'id' });
      }
      if (!db.objectStoreNames.contains('documents')) {
        const docStore = db.createObjectStore('documents', { keyPath: 'id' });
        docStore.createIndex('categoryId', 'categoryId', { unique: false });
      }
      if (!db.objectStoreNames.contains('chunks')) {
        const chunkStore = db.createObjectStore('chunks', { keyPath: 'id', autoIncrement: true });
        chunkStore.createIndex('documentId', 'documentId', { unique: false });
        chunkStore.createIndex('categoryId', 'categoryId', { unique: false });
      }
      if (!db.objectStoreNames.contains('settings')) {
        db.createObjectStore('settings', { keyPath: 'key' });
      }
    };
  });
};

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
    
    // Boost for rule references
    const rulePattern = /rule\s*\d+[-.]?\d*/gi;
    const queryRules = query.match(rulePattern) || [];
    const chunkRules = chunk.text.match(rulePattern) || [];
    queryRules.forEach(rule => {
      if (chunkRules.some(cr => cr.toLowerCase().includes(rule.toLowerCase()))) {
        score += 5;
      }
    });
    
    // Boost for exact phrase matches
    const queryLower = query.toLowerCase();
    if (chunk.text.toLowerCase().includes(queryLower)) {
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
  const [apiKey, setApiKey] = useState('');
  const [apiKeyInput, setApiKeyInput] = useState('');
  const [categories, setCategories] = useState([]);
  const [documents, setDocuments] = useState([]);
  const [chunks, setChunks] = useState([]);
  const [selectedCategory, setSelectedCategory] = useState(null);
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [activeTab, setActiveTab] = useState('chat');
  const [uploadProgress, setUploadProgress] = useState(null);
  const [newCategoryName, setNewCategoryName] = useState('');
  const [showSettings, setShowSettings] = useState(false);
  const messagesEndRef = useRef(null);

  // Load data from IndexedDB on mount
  useEffect(() => {
    loadData();
  }, []);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const loadData = async () => {
    try {
      const db = await openDB();
      
      // Load API key
      const settingsTransaction = db.transaction('settings', 'readonly');
      const settingsStore = settingsTransaction.objectStore('settings');
      const apiKeyRequest = settingsStore.get('apiKey');
      apiKeyRequest.onsuccess = () => {
        if (apiKeyRequest.result) {
          setApiKey(apiKeyRequest.result.value);
        }
      };
      
      // Load categories
      const catTransaction = db.transaction('categories', 'readonly');
      const catStore = catTransaction.objectStore('categories');
      const catRequest = catStore.getAll();
      catRequest.onsuccess = () => setCategories(catRequest.result || []);
      
      // Load documents
      const docTransaction = db.transaction('documents', 'readonly');
      const docStore = docTransaction.objectStore('documents');
      const docRequest = docStore.getAll();
      docRequest.onsuccess = () => setDocuments(docRequest.result || []);
      
      // Load chunks
      const chunkTransaction = db.transaction('chunks', 'readonly');
      const chunkStore = chunkTransaction.objectStore('chunks');
      const chunkRequest = chunkStore.getAll();
      chunkRequest.onsuccess = () => setChunks(chunkRequest.result || []);
    } catch (error) {
      console.error('Error loading data:', error);
    }
  };

  const saveApiKey = async () => {
    if (!apiKeyInput.trim()) return;
    
    try {
      const db = await openDB();
      const transaction = db.transaction('settings', 'readwrite');
      const store = transaction.objectStore('settings');
      store.put({ key: 'apiKey', value: apiKeyInput.trim() });
      setApiKey(apiKeyInput.trim());
      setApiKeyInput('');
      setShowSettings(false);
    } catch (error) {
      console.error('Error saving API key:', error);
    }
  };

  const clearApiKey = async () => {
    try {
      const db = await openDB();
      const transaction = db.transaction('settings', 'readwrite');
      const store = transaction.objectStore('settings');
      store.delete('apiKey');
      setApiKey('');
    } catch (error) {
      console.error('Error clearing API key:', error);
    }
  };

  const addCategory = async () => {
    if (!newCategoryName.trim()) return;
    
    const category = {
      id: Date.now().toString(),
      name: newCategoryName.trim(),
      createdAt: new Date().toISOString()
    };
    
    try {
      const db = await openDB();
      const transaction = db.transaction('categories', 'readwrite');
      transaction.objectStore('categories').add(category);
      setCategories([...categories, category]);
      setNewCategoryName('');
    } catch (error) {
      console.error('Error adding category:', error);
    }
  };

  const deleteCategory = async (categoryId) => {
    if (!confirm('Delete this category and all its documents?')) return;
    
    try {
      const db = await openDB();
      
      const catTransaction = db.transaction('categories', 'readwrite');
      catTransaction.objectStore('categories').delete(categoryId);
      
      const docsToDelete = documents.filter(d => d.categoryId === categoryId);
      const docTransaction = db.transaction('documents', 'readwrite');
      const docStore = docTransaction.objectStore('documents');
      docsToDelete.forEach(doc => docStore.delete(doc.id));
      
      const chunkTransaction = db.transaction('chunks', 'readwrite');
      const chunkStore = chunkTransaction.objectStore('chunks');
      chunks.filter(c => c.categoryId === categoryId).forEach(chunk => {
        if (chunk.id) chunkStore.delete(chunk.id);
      });
      
      setCategories(categories.filter(c => c.id !== categoryId));
      setDocuments(documents.filter(d => d.categoryId !== categoryId));
      setChunks(chunks.filter(c => c.categoryId !== categoryId));
      
      if (selectedCategory === categoryId) setSelectedCategory(null);
    } catch (error) {
      console.error('Error deleting category:', error);
    }
  };

  const extractTextFromPDF = async (file) => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = async (e) => {
        try {
          const typedArray = new Uint8Array(e.target.result);
          const pdf = await pdfjsLib.getDocument(typedArray).promise;
          let fullText = '';
          
          for (let i = 1; i <= pdf.numPages; i++) {
            setUploadProgress(`Extracting page ${i} of ${pdf.numPages}...`);
            const page = await pdf.getPage(i);
            const textContent = await page.getTextContent();
            const pageText = textContent.items.map(item => item.str).join(' ');
            fullText += `\n[Page ${i}]\n${pageText}\n`;
          }
          
          resolve(fullText);
        } catch (error) {
          reject(error);
        }
      };
      reader.onerror = reject;
      reader.readAsArrayBuffer(file);
    });
  };

  const uploadDocument = async (file, categoryId) => {
    if (!file || !categoryId) return;
    
    setUploadProgress('Starting upload...');
    
    try {
      const text = await extractTextFromPDF(file);
      
      const document = {
        id: Date.now().toString(),
        categoryId,
        name: file.name,
        uploadedAt: new Date().toISOString(),
        pageCount: (text.match(/\[Page \d+\]/g) || []).length
      };
      
      setUploadProgress('Saving document...');
      const db = await openDB();
      const docTransaction = db.transaction('documents', 'readwrite');
      docTransaction.objectStore('documents').add(document);
      
      setUploadProgress('Processing text chunks...');
      const textChunks = chunkText(text, 1500, 200);
      
      const chunkTransaction = db.transaction('chunks', 'readwrite');
      const chunkStore = chunkTransaction.objectStore('chunks');
      
      const newChunks = [];
      for (let i = 0; i < textChunks.length; i++) {
        setUploadProgress(`Indexing chunk ${i + 1} of ${textChunks.length}...`);
        const chunk = {
          documentId: document.id,
          categoryId,
          text: textChunks[i],
          index: i
        };
        chunkStore.add(chunk);
        newChunks.push(chunk);
      }
      
      setDocuments([...documents, document]);
      setChunks([...chunks, ...newChunks]);
      setUploadProgress(null);
      
    } catch (error) {
      console.error('Error uploading document:', error);
      setUploadProgress(`Error: ${error.message}`);
      setTimeout(() => setUploadProgress(null), 3000);
    }
  };

  const deleteDocument = async (documentId) => {
    if (!confirm('Delete this document?')) return;
    
    try {
      const db = await openDB();
      
      const docTransaction = db.transaction('documents', 'readwrite');
      docTransaction.objectStore('documents').delete(documentId);
      
      const chunkTransaction = db.transaction('chunks', 'readwrite');
      const chunkStore = chunkTransaction.objectStore('chunks');
      chunks.filter(c => c.documentId === documentId).forEach(chunk => {
        if (chunk.id) chunkStore.delete(chunk.id);
      });
      
      setDocuments(documents.filter(d => d.id !== documentId));
      setChunks(chunks.filter(c => c.documentId !== documentId));
    } catch (error) {
      console.error('Error deleting document:', error);
    }
  };

  const sendMessage = async () => {
    if (!input.trim() || isLoading || !selectedCategory || !apiKey) return;
    
    const userMessage = { role: 'user', content: input.trim() };
    const newMessages = [...messages, userMessage];
    setMessages(newMessages);
    setInput('');
    setIsLoading(true);
    
    try {
      const categoryChunks = chunks.filter(c => c.categoryId === selectedCategory);
      
      if (categoryChunks.length === 0) {
        setMessages([...newMessages, {
          role: 'assistant',
          content: "I don't have any documents loaded for this category yet. Please upload some PDFs first in the Manage tab."
        }]);
        setIsLoading(false);
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
      setIsLoading(false);
    }
  };

  const clearChat = () => {
    setMessages([]);
  };

  const getCategoryDocs = (categoryId) => documents.filter(d => d.categoryId === categoryId);
  const getCategoryChunkCount = (categoryId) => chunks.filter(c => c.categoryId === categoryId).length;

  // ============ RENDER ============
  return (
    <div className="flex flex-col h-screen bg-gray-100">
      {/* Header */}
      <div className="bg-gradient-to-r from-green-700 to-green-600 text-white p-4 shadow-lg">
        <div className="max-w-6xl mx-auto flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold flex items-center gap-2">
              ⚾ Baseball Rules Assistant
            </h1>
            <p className="text-green-200 text-sm mt-1">Upload rule books • Organize by category • Ask questions</p>
          </div>
          <button
            onClick={() => setShowSettings(!showSettings)}
            className="bg-green-600 hover:bg-green-500 px-4 py-2 rounded-lg text-sm font-medium transition-colors flex items-center gap-2"
          >
            ⚙️ Settings
          </button>
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
                Your API key is stored locally in your browser and never sent anywhere except Anthropic's API.
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
                
                {isLoading && (
                  <div className="flex justify-start">
                    <div className="bg-white shadow-md border border-gray-200 rounded-2xl rounded-bl-md px-5 py-3">
                      <div className="flex items-center gap-1 text-gray-500">
                        <span className="animate-pulse-dot">●</span>
                        <span className="animate-pulse-dot">●</span>
                        <span className="animate-pulse-dot">●</span>
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
                  disabled={!selectedCategory || isLoading || !apiKey}
                  className="flex-1 border border-gray-300 rounded-xl px-4 py-3 focus:outline-none focus:ring-2 focus:ring-green-500 disabled:bg-gray-100"
                />
                <button
                  onClick={sendMessage}
                  disabled={!selectedCategory || isLoading || !input.trim() || !apiKey}
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
