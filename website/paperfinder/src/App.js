import React, { useState, useEffect } from "react";
import axios from "axios";

function App() {
  // State for managing multiple conversations
  const [conversations, setConversations] = useState([
    { 
      id: 1, 
      name: "New Chat", 
      messages: [{ role: "bot", message: "Hi! I'm your research assistant. Ask me about any topic.", sources: [] }], 
      loading: false, 
      mode: "find_papers",
      filters: {
        yearStart: null,
        yearEnd: null,
        authors: [],
        venues: [],
        queryType: null
      }
    }
  ]);
  const [activeConversationId, setActiveConversationId] = useState(1);
  const [query, setQuery] = useState("");
  const [filtersOpen, setFiltersOpen] = useState(true);

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
      loading: false,
      mode: "find_papers", // Default mode
      filters: {
        yearStart: null,
        yearEnd: null,
        authors: [],
        venues: [],
        queryType: null
      }
    };
    setConversations([...conversations, newConversation]);
    setActiveConversationId(newId);
  };

  // Update filters for active conversation
  const updateFilters = (filterUpdates) => {
    setConversations(convs =>
      convs.map(c => 
        c.id === activeConversationId 
          ? { ...c, filters: { ...c.filters, ...filterUpdates } }
          : c
      )
    );
  };

  // Check if conversation is new (has no user messages yet)
  const isNewConversation = (conv) => {
    return conv.messages.filter(m => m.role === "user").length === 0;
  };

  // Set conversation mode (only allowed for new conversations)
  const setConversationMode = (conversationId, mode) => {
    const conv = conversations.find(c => c.id === conversationId);
    if (conv && isNewConversation(conv)) {
      setConversations(convs =>
        convs.map(c => c.id === conversationId ? { ...c, mode } : c)
      );
    }
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
    
    // Mode is now locked after first message (no need to explicitly set it, it's already set)
    
    setQuery("");
    
    // Set loading state for this conversation
    setConversations(convs =>
      convs.map(c => c.id === activeConversationId ? { ...c, loading: true } : c)
    );

    try {
      // Step 1: Analyze query immediately and populate filters
      let updatedFiltersFromAnalysis = null;
      try {
        const analyzeRes = await fetch("http://127.0.0.1:5000/api/analyze", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ query: currentQuery }),
        });
        
        if (analyzeRes.ok) {
          const analysisData = await analyzeRes.json();
          const analysis = analysisData.analysis || analysisData;
          
          // Immediately update filters from analysis
          if (analysis && analysis.status === "success") {
            // Get current filters
            const currentConv = conversations.find(c => c.id === activeConversationId);
            let updatedFilters = currentConv ? { ...currentConv.filters } : { ...activeConversation.filters };
            
            const timeRange = analysis.time_range || {};
            const authors = analysis.authors || [];
            const venues = analysis.venues || [];
            const queryType = analysis.query_type || null;
            
            // Update filters with extracted values (only if they exist in the analysis)
            if (timeRange.start !== null && timeRange.start !== undefined) {
              updatedFilters.yearStart = timeRange.start;
            }
            if (timeRange.end !== null && timeRange.end !== undefined) {
              updatedFilters.yearEnd = timeRange.end;
            }
            if (authors.length > 0) {
              // Merge with existing authors, avoiding duplicates
              const existingAuthors = updatedFilters.authors || [];
              const newAuthors = authors.filter(a => !existingAuthors.includes(a));
              updatedFilters.authors = [...existingAuthors, ...newAuthors];
            }
            if (venues.length > 0) {
              // Merge with existing venues, avoiding duplicates
              const existingVenues = updatedFilters.venues || [];
              const newVenues = venues.filter(v => !existingVenues.includes(v));
              updatedFilters.venues = [...existingVenues, ...newVenues];
            }
            if (queryType) {
              updatedFilters.queryType = queryType;
            }
            
            // Store for use in query
            updatedFiltersFromAnalysis = updatedFilters;
            
            // Update state immediately
            setConversations(convs =>
              convs.map(c => {
                if (c.id === activeConversationId) {
                  return { ...c, filters: updatedFilters };
                }
                return c;
              })
            );
          }
        }
      } catch (analyzeErr) {
        console.error("Error analyzing query:", analyzeErr);
        // Continue with the query even if analysis fails
      }
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

      // Step 2: Send query with current filters
      // Use filters updated from analysis, or fall back to current conversation filters
      const filtersToUse = updatedFiltersFromAnalysis || activeConversation.filters;
      const filtersPayload = {
        yearStart: filtersToUse.yearStart,
        yearEnd: filtersToUse.yearEnd,
        authors: filtersToUse.authors,
        venues: filtersToUse.venues,
        queryType: filtersToUse.queryType
      };
      
      const res = await fetch("http://127.0.0.1:5000/api/chat_stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ 
          message: currentQuery, 
          conversation_history: conversationHistory,
          filters: filtersPayload
        }),
      });

      if (!res.body) throw new Error("No response body");
      const reader = res.body.getReader();
      const decoder = new TextDecoder("utf-8");
      let buffer = "";
      let fullText = "";
      let finalSources = [];
      let queryAnalysis = null;

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
                queryAnalysis = payload.analysis || null;
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
      // Note: Filters are already updated immediately after analysis, so we don't need to update them again here
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

  const showModeTabs = isNewConversation(activeConversation);

  return (
    <div style={styles.container}>
      {/* Left Sidebar - Conversations */}
      <div style={styles.sidebar}>
        <div style={styles.sidebarHeader}>
          <h2 style={styles.sidebarTitle}>Conversations</h2>
          <button
            style={styles.newChatButtonSidebar}
            onClick={createNewConversation}
            title="New Chat"
          >
            + New Chat
          </button>
        </div>
        <div style={styles.conversationsList}>
          {conversations.map(conv => (
            <div
              key={conv.id}
              style={{
                ...styles.conversationItem,
                ...(conv.id === activeConversationId ? styles.conversationItemActive : {})
              }}
              onClick={() => setActiveConversationId(conv.id)}
            >
              <span style={styles.conversationName}>{conv.name}</span>
              {conversations.length > 1 && (
                <button
                  style={styles.conversationDelete}
                  onClick={(e) => deleteConversation(conv.id, e)}
                  onMouseEnter={(e) => {
                    e.target.style.backgroundColor = "#fee";
                    e.target.style.color = "#c00";
                  }}
                  onMouseLeave={(e) => {
                    e.target.style.backgroundColor = "transparent";
                    e.target.style.color = "#999";
                  }}
                  title="Delete conversation"
                >
                  ×
                </button>
              )}
            </div>
          ))}
        </div>
      </div>

      {/* Center - Main Chat Area */}
      <div style={styles.mainContent}>
        {/* Mode Tabs - Only show for new conversations */}
        {showModeTabs && (
          <div style={styles.modeTabsContainer}>
            <button
              style={{
                ...styles.modeTab,
                ...(activeConversation.mode === "find_papers" ? styles.modeTabActive : {})
              }}
              onClick={() => setConversationMode(activeConversationId, "find_papers")}
            >
              Find Papers
            </button>
            <button
              style={{
                ...styles.modeTab,
                ...(activeConversation.mode === "generate_report" ? styles.modeTabActive : {})
              }}
              onClick={() => setConversationMode(activeConversationId, "generate_report")}
            >
              Generate Report
            </button>
          </div>
        )}

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
            placeholder={
              activeConversation.mode === "generate_report" 
                ? "Describe what you'd like in your report..." 
                : "Ask about a topic (e.g. diffusion models for video)..."
            }
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

      {/* Right Sidebar - Filters */}
      <div style={{
        ...styles.filterSidebar,
        width: filtersOpen ? "300px" : "0px",
        border: filtersOpen ? "1px solid #e0e0e0" : "none",
        borderLeft: filtersOpen ? "none" : "none",
      }}>
        <div style={styles.filterHeader}>
          <h2 style={styles.filterTitle}>Filters</h2>
        </div>
        <div style={styles.filterContent}>
          {/* Year Range */}
          <div style={styles.filterSection}>
            <label style={styles.filterLabel}>Year Range</label>
            <div style={styles.yearInputs}>
              <input
                type="number"
                placeholder="Start"
                value={activeConversation.filters.yearStart || ""}
                onChange={(e) => updateFilters({ yearStart: e.target.value ? parseInt(e.target.value) : null })}
                style={styles.yearInput}
                min="1800"
                max="2100"
              />
              <span style={styles.yearSeparator}>-</span>
              <input
                type="number"
                placeholder="End"
                value={activeConversation.filters.yearEnd || ""}
                onChange={(e) => updateFilters({ yearEnd: e.target.value ? parseInt(e.target.value) : null })}
                style={styles.yearInput}
                min="1800"
                max="2100"
              />
            </div>
          </div>

          {/* Authors */}
          <div style={styles.filterSection}>
            <label style={styles.filterLabel}>Authors</label>
            <div style={styles.tagInputContainer}>
              {activeConversation.filters.authors.map((author, idx) => (
                <div key={idx} style={styles.tag}>
                  {author}
                  <button
                    onClick={() => {
                      const newAuthors = activeConversation.filters.authors.filter((_, i) => i !== idx);
                      updateFilters({ authors: newAuthors });
                    }}
                    style={styles.tagRemove}
                  >
                    ×
                  </button>
                </div>
              ))}
              <input
                type="text"
                placeholder="Add author..."
                onKeyDown={(e) => {
                  if (e.key === "Enter" && e.target.value.trim()) {
                    updateFilters({
                      authors: [...activeConversation.filters.authors, e.target.value.trim()]
                    });
                    e.target.value = "";
                  }
                }}
                style={styles.tagInput}
              />
            </div>
          </div>

          {/* Venues */}
          <div style={styles.filterSection}>
            <label style={styles.filterLabel}>Venues</label>
            <div style={styles.tagInputContainer}>
              {activeConversation.filters.venues.map((venue, idx) => (
                <div key={idx} style={styles.tag}>
                  {venue}
                  <button
                    onClick={() => {
                      const newVenues = activeConversation.filters.venues.filter((_, i) => i !== idx);
                      updateFilters({ venues: newVenues });
                    }}
                    style={styles.tagRemove}
                  >
                    ×
                  </button>
                </div>
              ))}
              <input
                type="text"
                placeholder="Add venue..."
                onKeyDown={(e) => {
                  if (e.key === "Enter" && e.target.value.trim()) {
                    updateFilters({
                      venues: [...activeConversation.filters.venues, e.target.value.trim()]
                    });
                    e.target.value = "";
                  }
                }}
                style={styles.tagInput}
              />
            </div>
          </div>

          {/* Query Type */}
          <div style={styles.filterSection}>
            <label style={styles.filterLabel}>Query Type</label>
            <select
              value={activeConversation.filters.queryType || ""}
              onChange={(e) => updateFilters({ queryType: e.target.value || null })}
              style={styles.selectInput}
            >
              <option value="">Any</option>
              <option value="BROAD_BY_DESCRIPTION">Broad by Description</option>
              <option value="SPECIFIC_BY_TITLE">Specific by Title</option>
              <option value="SPECIFIC_BY_NAME">Specific by Name</option>
              <option value="BY_AUTHOR">By Author</option>
            </select>
          </div>

          {/* Clear Filters Button */}
          <button
            onClick={() => updateFilters({
              yearStart: null,
              yearEnd: null,
              authors: [],
              venues: [],
              queryType: null
            })}
            style={styles.clearFiltersButton}
          >
            Clear All Filters
          </button>
        </div>
      </div>

      {/* Filter Toggle Button - Always visible */}
      <button
        onClick={() => setFiltersOpen(!filtersOpen)}
        style={{
          ...styles.filterToggleButton,
          right: filtersOpen ? "calc(300px + 20px)" : "20px"
        }}
        title={filtersOpen ? "Close filters" : "Open filters"}
      >
        {filtersOpen ? "◀" : "▶"}
      </button>
    </div>
  );
}

const styles = {
  container: {
    fontFamily: "Inter, sans-serif",
    backgroundColor: "#e6f2ef",
    height: "100vh",
    display: "flex",
    flexDirection: "row",
    overflow: "hidden",
  },
  sidebar: {
    width: "280px",
    backgroundColor: "#ffffff",
    borderRight: "1px solid #e0e0e0",
    display: "flex",
    flexDirection: "column",
    height: "100vh",
    overflow: "hidden",
  },
  sidebarHeader: {
    padding: "1rem",
    borderBottom: "1px solid #e0e0e0",
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
  },
  sidebarTitle: {
    color: "#156f63",
    fontSize: "1.2rem",
    margin: 0,
    fontWeight: "bold",
  },
  newChatButtonSidebar: {
    padding: "0.4rem 0.8rem",
    borderRadius: "6px",
    border: "1px solid #1c776a",
    backgroundColor: "#1c776a",
    color: "white",
    fontWeight: "bold",
    cursor: "pointer",
    fontSize: "0.85rem",
    transition: "all 0.2s",
  },
  newChatButtonSidebarHover: {
    backgroundColor: "#156f63",
  },
  conversationsList: {
    flex: 1,
    overflowY: "auto",
    padding: "0.5rem",
  },
  conversationItem: {
    padding: "0.75rem",
    borderRadius: "8px",
    marginBottom: "0.5rem",
    cursor: "pointer",
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
    backgroundColor: "#f9f9f9",
    transition: "all 0.2s",
  },
  conversationItemActive: {
    backgroundColor: "#e6f2ef",
    border: "2px solid #1c776a",
  },
  conversationName: {
    fontSize: "0.9rem",
    color: "#333",
    flex: 1,
    overflow: "hidden",
    textOverflow: "ellipsis",
    whiteSpace: "nowrap",
  },
  conversationDelete: {
    background: "none",
    border: "none",
    cursor: "pointer",
    fontSize: "1.2rem",
    color: "#999",
    padding: "0 0.25rem",
    borderRadius: "4px",
    transition: "all 0.2s",
  },
  mainContent: {
    flex: 1,
    display: "flex",
    flexDirection: "column",
    height: "100vh",
    padding: "1.2rem",
    paddingBottom: "1rem",
    overflow: "hidden",
    boxSizing: "border-box",
  },
  modeTabsContainer: {
    display: "flex",
    gap: "0.5rem",
    marginBottom: "1rem",
    justifyContent: "center",
    flexShrink: 0,
  },
  modeTab: {
    padding: "0.6rem 1.5rem",
    borderRadius: "8px",
    border: "2px solid #a4ccc1",
    backgroundColor: "#ffffff",
    color: "#1c776a",
    fontWeight: "bold",
    cursor: "pointer",
    fontSize: "0.95rem",
    transition: "all 0.2s",
  },
  modeTabActive: {
    backgroundColor: "#1c776a",
    borderColor: "#1c776a",
    color: "white",
  },
  chatBox: {
    flex: 1,
    width: "100%",
    backgroundColor: "#ffffff",
    borderRadius: "12px",
    padding: "1rem",
    overflowY: "auto",
    boxShadow: "0 4px 10px rgba(0,0,0,0.08)",
    display: "flex",
    flexDirection: "column",
    minHeight: 0,
    marginBottom: "0.8rem",
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
    flexShrink: 0,
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
  filterSidebar: {
    backgroundColor: "#ffffff",
    border: "1px solid #e0e0e0",
    borderLeft: "none",
    display: "flex",
    flexDirection: "column",
    height: "100vh",
    overflow: "hidden",
    borderRadius: "0 12px 0 0",
    boxShadow: "-2px 0 10px rgba(0,0,0,0.05)",
    flexShrink: 0,
    transition: "width 0.3s ease",
  },
  filterHeader: {
    padding: "1rem",
    borderBottom: "1px solid #e0e0e0",
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
  },
  filterCloseButton: {
    background: "none",
    border: "none",
    color: "#666",
    cursor: "pointer",
    fontSize: "1.5rem",
    lineHeight: "1",
    padding: "0",
    width: "24px",
    height: "24px",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    borderRadius: "50%",
    transition: "all 0.2s",
  },
  filterTitle: {
    color: "#156f63",
    fontSize: "1.2rem",
    margin: 0,
    fontWeight: "bold",
  },
  filterContent: {
    flex: 1,
    overflowY: "auto",
    overflowX: "hidden",
    padding: "1rem",
  },
  filterSection: {
    marginBottom: "1.5rem",
  },
  filterLabel: {
    display: "block",
    fontSize: "0.9rem",
    fontWeight: "bold",
    color: "#333",
    marginBottom: "0.5rem",
  },
  yearInputs: {
    display: "flex",
    alignItems: "center",
    gap: "0.5rem",
  },
  yearInput: {
    flex: 1,
    padding: "0.5rem",
    borderRadius: "6px",
    border: "1px solid #a4ccc1",
    fontSize: "0.9rem",
    outline: "none",
  },
  yearSeparator: {
    color: "#666",
    fontSize: "0.9rem",
  },
  tagInputContainer: {
    display: "flex",
    flexDirection: "column",
    gap: "0.5rem",
  },
  tag: {
    display: "inline-flex",
    alignItems: "center",
    gap: "0.5rem",
    padding: "0.4rem 0.6rem",
    backgroundColor: "#e6f2ef",
    borderRadius: "6px",
    fontSize: "0.85rem",
    color: "#1c776a",
  },
  tagRemove: {
    background: "none",
    border: "none",
    color: "#1c776a",
    cursor: "pointer",
    fontSize: "1.2rem",
    lineHeight: "1",
    padding: 0,
    width: "18px",
    height: "18px",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    borderRadius: "50%",
    transition: "background-color 0.2s",
  },
  tagInput: {
    width: "100%",
    padding: "0.5rem",
    borderRadius: "6px",
    border: "1px solid #a4ccc1",
    fontSize: "0.9rem",
    outline: "none",
  },
  selectInput: {
    width: "100%",
    padding: "0.5rem",
    borderRadius: "6px",
    border: "1px solid #a4ccc1",
    fontSize: "0.9rem",
    outline: "none",
    backgroundColor: "white",
    cursor: "pointer",
  },
  clearFiltersButton: {
    width: "100%",
    padding: "0.6rem",
    borderRadius: "6px",
    border: "1px solid #a4ccc1",
    backgroundColor: "#f4f9f7",
    color: "#1c776a",
    fontWeight: "bold",
    cursor: "pointer",
    fontSize: "0.9rem",
    transition: "all 0.2s",
    marginTop: "1rem",
  },
  filterToggleButton: {
    position: "fixed",
    top: "50%",
    transform: "translateY(-50%)",
    padding: "0.8rem 0.5rem",
    borderRadius: "8px 0 0 8px",
    border: "1px solid #1c776a",
    borderRight: "none",
    backgroundColor: "#1c776a",
    color: "white",
    fontWeight: "bold",
    cursor: "pointer",
    fontSize: "1rem",
    transition: "all 0.2s",
    zIndex: 100,
    boxShadow: "-2px 0 10px rgba(0,0,0,0.1)",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    width: "30px",
    height: "60px",
  },
};

export default App;
