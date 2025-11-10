import React, { useState, useEffect } from "react";
import axios from "axios";

function App() {
  // State for managing multiple conversations
  const [conversations, setConversations] = useState([
    { id: 1, name: "New Chat", messages: [{ role: "bot", message: "Hi! I'm your research assistant. Ask me about any topic.", sources: [] }], loading: false }
  ]);
  const [activeConversationId, setActiveConversationId] = useState(1);
  const [query, setQuery] = useState("");

  // Get current conversation
  const activeConversation = conversations.find(c => c.id === activeConversationId) || conversations[0];
  const messages = activeConversation.messages;

  // Typeset LaTeX after messages render/update
  useEffect(() => {
    const typeset = async () => {
      try {
        if (window.MathJax && window.MathJax.typesetPromise) {
          await window.MathJax.typesetPromise();
        }
      } catch (e) {
        // no-op
      }
    };
    typeset();
  }, [messages]);

  // Create new conversation
  const createNewConversation = () => {
    const newId = Math.max(...conversations.map(c => c.id), 0) + 1;
    const newConversation = {
      id: newId,
      name: `Chat ${newId}`,
      messages: [{ role: "bot", message: "Hi! I'm your research assistant. Ask me about any topic.", sources: [] }],
      loading: false
    };
    setConversations([...conversations, newConversation]);
    setActiveConversationId(newId);
  };

  // Delete conversation
  const deleteConversation = (id, e) => {
    e.stopPropagation(); // Prevent tab switching
    if (conversations.length === 1) return; // Don't delete the last conversation
    
    const filtered = conversations.filter(c => c.id !== id);
    setConversations(filtered);
    
    // If deleted conversation was active, switch to first available
    if (id === activeConversationId) {
      setActiveConversationId(filtered[0].id);
    }
  };

  // Update conversation name based on first user message
  const updateConversationName = (conversationId, newName) => {
    setConversations(convs => 
      convs.map(c => c.id === conversationId ? { ...c, name: newName } : c)
    );
  };

  const handleSend = async () => {
    if (!query.trim()) return;
    const userMessage = { role: "user", message: query, sources: [] };
    const currentQuery = query;
    
    // Update conversation with user message
    const updatedMessages = [...messages, userMessage];
    setConversations(convs =>
      convs.map(c => c.id === activeConversationId ? { ...c, messages: updatedMessages } : c)
    );
    
    // Update conversation name if it's still default
    if (activeConversation.name === "New Chat" || activeConversation.name.startsWith("Chat ")) {
      const newName = currentQuery.length > 30 ? currentQuery.substring(0, 30) + "..." : currentQuery;
      updateConversationName(activeConversationId, newName);
    }
    
    setQuery("");
    
    // Set loading state for this conversation
    setConversations(convs =>
      convs.map(c => c.id === activeConversationId ? { ...c, loading: true } : c)
    );

    try {
      // Build conversation history from previous messages (excluding the greeting)
      const conversationHistory = updatedMessages
        .filter(msg => msg.role !== "bot" || msg.message !== "Hi! I'm your research assistant. Ask me about any topic.")
        .slice(-6)
        .map(msg => ({
          role: msg.role === "bot" ? "assistant" : "user",
          content: msg.message
        }));
      
      // Insert a placeholder bot message we will update progressively
      const placeholderIndex = updatedMessages.length; // next index
      setConversations(convs =>
        convs.map(c => c.id === activeConversationId ? { ...c, messages: [...updatedMessages, { role: "bot", message: "", sources: [] }] } : c)
      );

      const res = await fetch("http://127.0.0.1:5000/api/chat_stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: currentQuery, conversation_history: conversationHistory }),
      });

      if (!res.body) throw new Error("No response body");
      const reader = res.body.getReader();
      const decoder = new TextDecoder("utf-8");
      let buffer = "";
      let fullText = "";
      let finalSources = [];

      const processBuffer = (buf) => {
        const lines = buf.split("\n\n");
        for (let i = 0; i < lines.length - 1; i++) {
          const line = lines[i].trim();
          if (line.startsWith("data: ")) {
            try {
              const payload = JSON.parse(line.slice(6));
              if (payload.event === "delta" && payload.text) {
                fullText += payload.text;
                // Update the last bot message progressively
                setConversations(convs => convs.map(c => {
                  if (c.id !== activeConversationId) return c;
                  const newMessages = [...c.messages];
                  newMessages[placeholderIndex] = { role: "bot", message: fullText, sources: [] };
                  return { ...c, messages: newMessages };
                }));
              } else if (payload.event === "done") {
                finalSources = payload.sources || [];
              }
            } catch {}
          }
        }
        return lines[lines.length - 1];
      };

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        buffer = processBuffer(buffer);
      }
      // Flush remaining buffer
      if (buffer) buffer = processBuffer(buffer + "\n\n");

      // Finalize: set sources and clear loading
      setConversations(convs =>
        convs.map(c => {
          if (c.id !== activeConversationId) return c;
          const newMessages = [...c.messages];
          newMessages[placeholderIndex] = { role: "bot", message: fullText, sources: finalSources };
          return { ...c, messages: newMessages, loading: false };
        })
      );
    } catch (err) {
      const errorMessage = { role: "bot", message: "⚠️ Backend not reachable. Make sure Flask is running.", sources: [] };
      // Update conversation with error message and remove loading state
      setConversations(convs =>
        convs.map(c => c.id === activeConversationId ? { ...c, messages: [...updatedMessages, errorMessage], loading: false } : c)
      );
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === "Enter" && !activeConversation.loading) handleSend();
  };

  return (
    <div style={styles.container}>
      {/* 🌿 Header */}
      <header style={styles.header}>
        <h1 style={styles.title}>Team One Paper Finder</h1>
        <p style={styles.subtitle}>Your AI-powered research assistant</p>
      </header>

      {/* Conversation Tabs */}
      <div style={styles.tabsContainer}>
        <div style={styles.tabs}>
          {conversations.map(conv => (
            <div
              key={conv.id}
              style={{
                ...styles.tab,
                ...(conv.id === activeConversationId ? styles.tabActive : {})
              }}
              onClick={() => setActiveConversationId(conv.id)}
            >
              <span style={styles.tabName}>{conv.name}</span>
              {conversations.length > 1 && (
                <button
                  style={{
                    ...styles.tabClose,
                    color: conv.id === activeConversationId ? "white" : "#666",
                  }}
                  onMouseEnter={(e) => {
                    if (conv.id === activeConversationId) {
                      e.target.style.backgroundColor = "rgba(255,255,255,0.2)";
                    } else {
                      e.target.style.backgroundColor = "#e0e0e0";
                    }
                  }}
                  onMouseLeave={(e) => {
                    e.target.style.backgroundColor = "transparent";
                  }}
                  onClick={(e) => deleteConversation(conv.id, e)}
                  title="Delete conversation"
                >
                  ×
                </button>
              )}
            </div>
          ))}
          <button
            style={styles.newChatButton}
            onClick={createNewConversation}
            title="New Chat"
          >
            + New
          </button>
        </div>
      </div>

      {/* Chat Window */}
      <div style={styles.chatBox}>
        {messages.map((m, i) => (
          <div
            key={i}
            style={{
              ...styles.message,
              alignSelf: m.role === "user" ? "flex-end" : "flex-start",
              backgroundColor: m.role === "user" ? "#d7f3eb" : "#f4f9f7",
            }}
          >
            {/* Main message content (simple formatting only) */}
            <p
              style={styles.text}
              dangerouslySetInnerHTML={{
                __html: m.message
                  .replace(/\*\*(.*?)\*\*/g, "<b>$1</b>")
                  .replace(/\*(.*?)\*/g, "<i>$1</i>")
                  .replace(/\n/g, "<br/>")
              }}
            />
            
            {/* Sources section */}
            {m.sources && m.sources.length > 0 && (
              <div style={styles.sources}>
                <div style={styles.sourcesTitle}>📚 Sources:</div>
                {m.sources.map((source, idx) => (
                  <div key={idx} style={styles.sourceItem}>
                    <span style={styles.sourceNumber}>{source.rank || idx + 1}.</span>
                    <span style={styles.sourceId}>[{source.paper_id}]</span>
                    {source.url ? (
                      <a href={source.url} target="_blank" rel="noopener noreferrer" style={{...styles.sourceTitle, color: "#1c776a", textDecoration: "underline"}}>
                        {source.title}
                      </a>
                    ) : (
                      <span style={styles.sourceTitle}>{source.title}</span>
                    )}
                    {source.authors && (
                      <span style={styles.sourceAuthors}> - {source.authors}</span>
                    )}
                  </div>
                ))}
              </div>
            )}
          </div>
        ))}
        {activeConversation.loading && !(messages.length > 0 && messages[messages.length - 1].role === "bot") && (
          <div style={{ ...styles.message, backgroundColor: "#f4f9f7" }}>
            <p style={styles.text}>Searching for papers...</p>
          </div>
        )}
      </div>

      {/* Input bar */}
      <div style={styles.inputBar}>
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={handleKeyPress}
          placeholder="Ask about a topic (e.g. diffusion models for video)..."
          style={styles.input}
          disabled={activeConversation.loading}
        />
        <button
          onClick={handleSend}
          disabled={activeConversation.loading}
          style={{ ...styles.button, opacity: activeConversation.loading ? 0.7 : 1 }}
        >
          {activeConversation.loading ? "..." : "Send"}
        </button>
      </div>
    </div>
  );
}

const styles = {
  container: {
    fontFamily: "Inter, sans-serif",
    backgroundColor: "#e6f2ef",
    height: "100vh",
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    padding: "1.2rem",
  },
  header: {
    textAlign: "center",
    marginBottom: "1rem",
  },
  title: {
    color: "#156f63",
    fontSize: "2rem",
    marginBottom: "0.3rem",
  },
  subtitle: {
    color: "#285c54",
    fontSize: "1rem",
    marginTop: 0,
  },
  chatBox: {
    flex: 1,
    width: "100%",
    maxWidth: "650px", // 🔹 narrower than before
    backgroundColor: "#ffffff",
    borderRadius: "12px",
    padding: "1rem",
    overflowY: "auto",
    boxShadow: "0 4px 10px rgba(0,0,0,0.08)",
    display: "flex",
    flexDirection: "column",
  },
  message: {
    margin: "0.4rem 0",
    padding: "0.7rem 1rem",
    borderRadius: "12px",
    maxWidth: "80%",
    lineHeight: 1.5,
    wordBreak: "break-word",
  },
  text: {
    margin: 0,
    whiteSpace: "pre-wrap",
  },
  inputBar: {
    display: "flex",
    width: "100%",
    maxWidth: "650px",
    marginTop: "0.8rem",
  },
  input: {
    flex: 1,
    borderRadius: "12px",
    border: "1px solid #a4ccc1",
    padding: "0.8rem",
    fontSize: "1rem",
    outline: "none",
  },
  button: {
    marginLeft: "10px",
    borderRadius: "12px",
    border: "none",
    backgroundColor: "#1c776a",
    color: "white",
    fontWeight: "bold",
    padding: "0.8rem 1.4rem",
    cursor: "pointer",
  },
  sources: {
    marginTop: "0.8rem",
    paddingTop: "0.8rem",
    borderTop: "1px solid #e0e0e0",
  },
  sourcesTitle: {
    fontWeight: "bold",
    color: "#285c54",
    marginBottom: "0.4rem",
  },
  sourceItem: {
    marginBottom: "0.4rem",
    fontSize: "0.9rem",
    color: "#333",
    lineHeight: "1.4",
    display: "block",
  },
  sourceNumber: {
    fontWeight: "bold",
    color: "#1c776a",
    marginRight: "0.4rem",
  },
  sourceId: {
    fontWeight: "bold",
    color: "#1c776a",
    marginRight: "0.5rem",
    whiteSpace: "nowrap",
  },
  sourceTitle: {
    color: "#555",
  },
  sourceAuthors: {
    color: "#777",
    fontSize: "0.85rem",
    fontStyle: "italic",
  },
  tabsContainer: {
    width: "100%",
    maxWidth: "650px",
    marginBottom: "0.8rem",
  },
  tabs: {
    display: "flex",
    gap: "0.5rem",
    overflowX: "auto",
    paddingBottom: "0.5rem",
    scrollbarWidth: "thin",
  },
  tab: {
    display: "flex",
    alignItems: "center",
    gap: "0.5rem",
    padding: "0.5rem 1rem",
    borderRadius: "8px",
    backgroundColor: "#ffffff",
    border: "2px solid #e0e0e0",
    cursor: "pointer",
    whiteSpace: "nowrap",
    transition: "all 0.2s",
  },
  tabActive: {
    backgroundColor: "#1c776a",
    borderColor: "#1c776a",
    color: "white",
  },
  tabName: {
    fontSize: "0.9rem",
    maxWidth: "150px",
    overflow: "hidden",
    textOverflow: "ellipsis",
  },
  tabClose: {
    background: "none",
    border: "none",
    cursor: "pointer",
    fontSize: "1.2rem",
    lineHeight: "1",
    padding: "0",
    width: "18px",
    height: "18px",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    borderRadius: "50%",
    transition: "background-color 0.2s",
  },
  newChatButton: {
    padding: "0.5rem 1rem",
    borderRadius: "8px",
    border: "2px dashed #a4ccc1",
    backgroundColor: "#f4f9f7",
    color: "#1c776a",
    fontWeight: "bold",
    cursor: "pointer",
    whiteSpace: "nowrap",
    transition: "all 0.2s",
  },
};

export default App;
