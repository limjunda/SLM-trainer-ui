import React, { useState, useEffect, useRef } from 'react';

// --- Helper Components ---

const Card = ({ children, className = '' }) => (
  <div className={`bg-white/10 border border-slate-700 rounded-2xl shadow-lg backdrop-blur-xl p-6 sm:p-8 ${className}`}>
    {children}
  </div>
);

const Button = ({ children, onClick, disabled = false, className = '' }) => (
  <button
    onClick={onClick}
    disabled={disabled}
    className={`w-full px-4 py-3 font-semibold text-white bg-indigo-600 rounded-lg hover:bg-indigo-700 disabled:bg-indigo-400 disabled:cursor-not-allowed focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 transition-all duration-200 ${className}`}
  >
    {children}
  </button>
);

const Input = ({ type = 'text', value, onChange, placeholder, disabled = false }) => (
  <input
    type={type}
    value={value}
    onChange={onChange}
    placeholder={placeholder}
    disabled={disabled}
    className="w-full px-4 py-3 bg-slate-800 border border-slate-600 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500 transition-colors"
  />
);

const Spinner = () => (
    <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white"></div>
);

const Alert = ({ message, type = 'info' }) => {
    const colors = {
        info: 'bg-blue-900/50 border-blue-500 text-blue-200',
        success: 'bg-green-900/50 border-green-500 text-green-200',
        error: 'bg-red-900/50 border-red-500 text-red-200',
    };
    if (!message) return null;
    return (
        <div className={`p-4 mt-4 text-sm rounded-lg border ${colors[type]}`} role="alert">
            <span className="font-medium">{message}</span>
        </div>
    );
};


// --- Main Application Components ---

const PageSwitcher = ({ currentPage, setCurrentPage }) => (
  <nav className="flex justify-center mb-8 space-x-4">
    <button
      onClick={() => setCurrentPage('chat')}
      className={`px-6 py-2 font-semibold rounded-lg transition-colors ${currentPage === 'chat' ? 'bg-indigo-600 text-white' : 'bg-slate-800/50 text-slate-300 hover:bg-slate-700/50'}`}
    >
      Chat & Inference
    </button>
    <button
      onClick={() => setCurrentPage('train')}
      className={`px-6 py-2 font-semibold rounded-lg transition-colors ${currentPage === 'train' ? 'bg-indigo-600 text-white' : 'bg-slate-800/50 text-slate-300 hover:bg-slate-700/50'}`}
    >
      Fine-Tuning
    </button>
  </nav>
);

const HuggingFaceAuth = ({ token, setToken, onSave, loading }) => (
  <Card>
    <h2 className="text-2xl font-bold text-white mb-4">Hugging Face Authentication</h2>
    <p className="text-slate-400 mb-4">Enter your Hugging Face token with 'write' access to download models and upload trained adapters.</p>
    <div className="flex space-x-2">
      <Input
        type="password"
        value={token}
        onChange={(e) => setToken(e.target.value)}
        placeholder="hf_..."
      />
      <Button onClick={onSave} disabled={loading || !token} className="w-auto flex items-center justify-center">
        {loading ? <Spinner /> : 'Save'}
      </Button>
    </div>
  </Card>
);

const ModelManager = ({ onModelLoaded }) => {
  const [query, setQuery] = useState('meta-llama/Llama-2-7b-chat-hf');
  const [searchResults, setSearchResults] = useState([]);
  const [loadingState, setLoadingState] = useState({ searching: false, loading: '' }); // loading is modelId
  const [error, setError] = useState('');

  const handleSearch = async () => {
    if (!query) return;
    setLoadingState({ ...loadingState, searching: true });
    setError('');
    try {
      const response = await fetch(`http://127.0.0.1:5000/search?query=${encodeURIComponent(query)}`);
      if (!response.ok) throw new Error('Failed to fetch models.');
      const data = await response.json();
      setSearchResults(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoadingState({ ...loadingState, searching: false });
    }
  };

  const handleLoad = async (modelId) => {
    setLoadingState({ ...loadingState, loading: modelId });
    setError('');
    try {
      const response = await fetch('http://127.0.0.1:5000/load-model', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model_id: modelId }),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || 'Failed to load model.');
      onModelLoaded(modelId);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoadingState({ ...loadingState, loading: '' });
    }
  };
  
  useEffect(() => {
     handleSearch(); // initial search on load
  }, []);

  return (
    <Card>
      <h2 className="text-2xl font-bold text-white mb-4">Search & Load Model</h2>
      <div className="flex space-x-2 mb-4">
        <Input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="e.g., meta-llama/Llama-2-7b-chat-hf"
        />
        <Button onClick={handleSearch} disabled={loadingState.searching} className="w-auto flex justify-center items-center">
            {loadingState.searching ? <Spinner/> : 'Search'}
        </Button>
      </div>
      <Alert message={error} type="error" />
      <div className="max-h-60 overflow-y-auto space-y-2 pr-2">
        {searchResults.map((model) => (
          <div key={model.id} className="bg-slate-800 p-3 rounded-lg flex justify-between items-center">
            <span className="text-slate-300 truncate font-mono text-sm">{model.id}</span>
            <Button onClick={() => handleLoad(model.id)} disabled={!!loadingState.loading} className="w-auto !py-2 text-sm flex justify-center items-center">
                {loadingState.loading === model.id ? <Spinner/> : 'Load'}
            </Button>
          </div>
        ))}
      </div>
    </Card>
  );
};

const Chatbot = ({ loadedModel }) => {
    const [messages, setMessages] = useState([]);
    const [input, setInput] = useState('');
    const [loading, setLoading] = useState(false);
    const messagesEndRef = useRef(null);

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    };

    useEffect(scrollToBottom, [messages]);
    
    useEffect(() => {
        if(loadedModel){
            setMessages([{ sender: 'bot', text: `Model '${loadedModel}' is ready! How can I help you?` }]);
        } else {
            setMessages([]);
        }
    }, [loadedModel]);


    const handleSend = async () => {
        if (!input.trim() || !loadedModel) return;

        const newMessages = [...messages, { sender: 'user', text: input }];
        setMessages(newMessages);
        setInput('');
        setLoading(true);

        try {
            const response = await fetch('http://127.0.0.1:5000/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ prompt: input, messages: messages }),
            });
            const data = await response.json();
             if (!response.ok) throw new Error(data.error || 'Failed to get response from model.');
            setMessages([...newMessages, { sender: 'bot', text: data.response }]);
        } catch (err) {
            setMessages([...newMessages, { sender: 'bot', text: `Error: ${err.message}` }]);
        } finally {
            setLoading(false);
        }
    };

    return (
        <Card className="flex flex-col h-[70vh]">
            <h2 className="text-2xl font-bold text-white mb-4">Chatbot</h2>
            <div className="flex-grow bg-slate-800 rounded-lg p-4 overflow-y-auto mb-4 space-y-4">
                 {!loadedModel ? (
                    <div className="flex items-center justify-center h-full text-slate-400">
                        Please load a model to start chatting.
                    </div>
                 ) : (
                    messages.map((msg, index) => (
                    <div key={index} className={`flex ${msg.sender === 'user' ? 'justify-end' : 'justify-start'}`}>
                        <div className={`max-w-xs md:max-w-md lg:max-w-lg px-4 py-2 rounded-2xl ${msg.sender === 'user' ? 'bg-indigo-600 text-white' : 'bg-slate-700 text-slate-200'}`}>
                            <p className="whitespace-pre-wrap">{msg.text}</p>
                        </div>
                    </div>
                    ))
                )}
                <div ref={messagesEndRef} />
            </div>
            <div className="flex space-x-2">
                <Input
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    placeholder={loadedModel ? "Type your message..." : "Load a model first"}
                    disabled={!loadedModel || loading}
                    onKeyDown={(e) => e.key === 'Enter' && handleSend()}
                />
                <Button onClick={handleSend} disabled={!loadedModel || loading} className="w-auto flex items-center justify-center">
                    {loading ? <Spinner/> : 'Send'}
                </Button>
            </div>
        </Card>
    );
};


const Trainer = () => {
    const [file, setFile] = useState(null);
    const [eda, setEda] = useState(null);
    const [status, setStatus] = useState('');
    const [logs, setLogs] = useState('');
    const [loading, setLoading] = useState(false);
    const [trainConfig, setTrainConfig] = useState({
        model_id: 'meta-llama/Llama-2-7b-chat-hf',
        new_model_name: 'my-finetuned-slm',
        framework: 'peft',
        epochs: '1',
    });
    
    const logRef = useRef(null);
    useEffect(() => {
        if(logRef.current) {
            logRef.current.scrollTop = logRef.current.scrollHeight;
        }
    }, [logs]);

    const handleFileChange = (e) => {
        setFile(e.target.files[0]);
        setEda(null);
        setStatus('');
    };

    const handleUpload = async () => {
        if (!file) return;
        const formData = new FormData();
        formData.append('dataset', file);
        setLoading(true);
        setStatus('Uploading and analyzing dataset...');
        try {
            const response = await fetch('http://127.0.0.1:5000/upload-dataset', {
                method: 'POST',
                body: formData,
            });
            const data = await response.json();
            if (!response.ok) throw new Error(data.error || 'Failed to process dataset.');
            setEda(data.eda);
            setStatus(`Dataset '${file.name}' loaded. Ready for training.`);
        } catch (err) {
            setStatus(`Error: ${err.message}`);
        } finally {
            setLoading(false);
        }
    };

    const handleTrain = async () => {
        setLoading(true);
        setLogs('');
        setStatus('Starting fine-tuning process... This may take a long time.');

        try {
            const response = await fetch('http://127.0.0.1:5000/train', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(trainConfig),
            });
            
            if (!response.body) {
                throw new Error("ReadableStream not supported or not available.");
            }

            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            
            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                const chunk = decoder.decode(value, { stream: true });
                setLogs(prev => prev + chunk);
            }

            setStatus('Training finished successfully! You can load the new model in the chat tab.');

        } catch (err) {
            setStatus(`Training Error: ${err.message}`);
            setLogs(prev => prev + `\n\n--- ERROR ---\n${err.message}`);
        } finally {
            setLoading(false);
        }
    };
    
    const handleConfigChange = (e) => {
        setTrainConfig({ ...trainConfig, [e.target.name]: e.target.value });
    };

    return (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            <div>
                <Card>
                    <h2 className="text-2xl font-bold text-white mb-4">1. Upload Dataset</h2>
                    <p className="text-slate-400 mb-4">Upload a CSV or JSON file. It should contain a 'text' column for training.</p>
                    <div className="flex items-center space-x-2">
                        <input type="file" onChange={handleFileChange} className="block w-full text-sm text-slate-400 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-indigo-50 file:text-indigo-700 hover:file:bg-indigo-100"/>
                        <Button onClick={handleUpload} disabled={!file || loading} className="w-auto flex items-center justify-center">
                           {loading ? <Spinner /> : 'Upload'}
                        </Button>
                    </div>
                </Card>
                
                {eda && (
                    <Card className="mt-8">
                        <h2 className="text-2xl font-bold text-white mb-4">Data EDA</h2>
                        <div className="space-y-2 text-slate-300">
                           <p><strong>Shape:</strong> {eda.shape}</p>
                           <p><strong>Columns:</strong> {eda.columns.join(', ')}</p>
                           <p><strong>Sample Rows:</strong></p>
                           <pre className="bg-slate-800 p-2 rounded-md text-xs overflow-x-auto">{eda.head}</pre>
                        </div>
                    </Card>
                )}

                <Card className="mt-8">
                    <h2 className="text-2xl font-bold text-white mb-4">2. Configure & Train</h2>
                    <div className="space-y-4">
                        <div>
                            <label className="block text-sm font-medium text-slate-300 mb-1">Base Model</label>
                            <Input name="model_id" value={trainConfig.model_id} onChange={handleConfigChange} placeholder="e.g., meta-llama/Llama-2-7b-chat-hf"/>
                        </div>
                        <div>
                            <label className="block text-sm font-medium text-slate-300 mb-1">New Model Name (for Hugging Face)</label>
                            <Input name="new_model_name" value={trainConfig.new_model_name} onChange={handleConfigChange} placeholder="e.g., my-finetuned-llama"/>
                        </div>
                        <div>
                             <label className="block text-sm font-medium text-slate-300 mb-1">Training Framework</label>
                             <select name="framework" value={trainConfig.framework} onChange={handleConfigChange} className="w-full px-4 py-3 bg-slate-800 border border-slate-600 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500 transition-colors">
                                <option value="peft">PEFT (LoRA) - Recommended</option>
                                <option value="full" disabled>Full Fine-tuning (Requires High VRAM)</option>
                             </select>
                        </div>
                        <div>
                             <label className="block text-sm font-medium text-slate-300 mb-1">Epochs</label>
                             <Input name="epochs" type="number" value={trainConfig.epochs} onChange={handleConfigChange} placeholder="1"/>
                        </div>
                        <Button onClick={handleTrain} disabled={!eda || loading}>
                            {loading ? 'Training...' : 'Start Training'}
                        </Button>
                    </div>
                </Card>
            </div>
            
            <Card>
                <h2 className="text-2xl font-bold text-white mb-4">3. Training Status & Logs</h2>
                <div className="bg-slate-800 p-4 rounded-lg h-[80vh] flex flex-col">
                    <p className="text-slate-400 mb-2 flex-shrink-0">{status}</p>
                    <pre ref={logRef} className="text-xs text-slate-300 font-mono overflow-y-auto h-full w-full whitespace-pre-wrap break-words flex-grow">
                        {logs || (loading ? "Waiting for training to start..." : "Logs will appear here.")}
                    </pre>
                </div>
            </Card>
        </div>
    );
};


export default function App() {
  const [hfToken, setHfToken] = useState('');
  const [tokenSaved, setTokenSaved] = useState(false);
  const [authLoading, setAuthLoading] = useState(false);
  const [authError, setAuthError] = useState('');
  const [loadedModel, setLoadedModel] = useState('');
  const [currentPage, setCurrentPage] = useState('chat'); // 'chat' or 'train'
  
  const handleSaveToken = async () => {
    setAuthLoading(true);
    setAuthError('');
    try {
        const response = await fetch('http://127.0.0.1:5000/set-token', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ token: hfToken }),
        });
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || 'Failed to authenticate.');
        setTokenSaved(true);
    } catch(err) {
        setAuthError(err.message);
        setTokenSaved(false);
    } finally {
        setAuthLoading(false);
    }
  };

  return (
    <div className="bg-slate-900 text-slate-200 min-h-screen font-sans p-4 sm:p-6 lg:p-8">
      <div className="absolute top-0 left-0 w-full h-full bg-grid-slate-700/[0.05]"></div>
      <div className="max-w-screen-2xl mx-auto relative">
        <header className="text-center mb-10">
          <h1 className="text-4xl sm:text-5xl font-extrabold text-white tracking-tight">
            SLM Studio
          </h1>
          <p className="mt-2 text-lg text-slate-400">Interact with, fine-tune, and deploy Small Language Models locally.</p>
        </header>

        <PageSwitcher currentPage={currentPage} setCurrentPage={setCurrentPage} />
        
        {currentPage === 'chat' && (
             <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 items-start">
                <div className="space-y-8">
                   <HuggingFaceAuth token={hfToken} setToken={setHfToken} onSave={handleSaveToken} loading={authLoading} />
                   {authError && <Alert message={authError} type="error" />}
                   {tokenSaved && <Alert message="Token saved successfully! You can now search and load models." type="success" />}
                   
                   {tokenSaved && <ModelManager onModelLoaded={setLoadedModel} />}
                </div>
                <div>
                   <Chatbot loadedModel={loadedModel} />
                </div>
             </div>
        )}
        
        {currentPage === 'train' && (
            <div>
               {!tokenSaved && (
                <Card className="max-w-md mx-auto">
                    <h2 className="text-2xl font-bold text-center mb-4">Authentication Required</h2>
                    <p className="text-slate-400 text-center">Please go to the 'Chat & Inference' page to set your Hugging Face token first. A token is required to download base models and upload your fine-tuned versions.</p>
                </Card>
               )}
               {tokenSaved && <Trainer />}
            </div>
        )}

      </div>
    </div>
  );
}
