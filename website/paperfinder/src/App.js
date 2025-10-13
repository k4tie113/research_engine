import React, { useState } from "react";
import axios from "axios";

function App() {
  const [messages, setMessages] = useState([
    { role: "bot", message: "Hi! I’m your research assistant. Ask me about any topic." },
  ]);
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(false);

  const handleSend = async () => {
    if (!query.trim()) return;
    const userMessage = { role: "user", message: query };
    setMessages((prev) => [...prev, userMessage]);
    setQuery("");
    setLoading(true);

    try {
      const res = await axios.post("http://127.0.0.1:5000/api/chat", { message: query });
      const botMessage = { role: "bot", message: res.data.reply };

      setMessages((prev) => [...prev, botMessage]);
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        { role: "bot", message: "⚠️ Backend not reachable. Make sure Flask is running." },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === "Enter" && !loading) handleSend();
  };

  return (
    <div style={styles.container}>
      {/* 🌿 Header */}
      <header style={styles.header}>
        <h1 style={styles.title}>Team One Paper Finder</h1>
        <p style={styles.subtitle}>Your AI-powered research assistant</p>
      </header>

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
            <p
            style={styles.text}
            dangerouslySetInnerHTML={{
            __html: m.message
            .replace(/\*\*(.*?)\*\*/g, "<b>$1</b>")        // bold
            .replace(/\n/g, "<br/>")                       // line breaks
            .replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank" style="color:#1c776a;text-decoration:underline;">$1</a>'), // clickable links
          }}
          />

          </div>
        ))}
        {loading && (
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
          disabled={loading}
        />
        <button
          onClick={handleSend}
          disabled={loading}
          style={{ ...styles.button, opacity: loading ? 0.7 : 1 }}
        >
          {loading ? "..." : "Send"}
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
};

export default App;
