import axios from "axios";
import React, { useState } from "react";

function App() {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState([]);
  const [expanded, setExpanded] = useState({});
  const [loading, setLoading] = useState(false); // NEW STATE

  const handleSearch = async () => {
    if (!query.trim()) return;
    setLoading(true); // ⏳ Start loading
    try {
      const res = await axios.get(`http://127.0.0.1:5000/api/search?q=${query}`);
      setResults(res.data.results);
    } catch (e) {
      alert("Backend not reachable. Make sure Flask is running on port 5000.");
    } finally {
      setLoading(false); // Stop loading
    }
  };

  const toggleAbstract = (i) => {
    setExpanded((prev) => ({ ...prev, [i]: !prev[i] }));
  };

  return (
    <div style={styles.container}>
      <header style={styles.header}>
        <h1 style={styles.title}>🌿 Paper Finder – Team One</h1>
        <p style={styles.subtitle}>
          A scholarly search prototype — <b>powered by Semantic Scholar</b> (FAISS integration coming soon)
        </p>
      </header>

      <div style={styles.searchBox}>
        <input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Describe the papers you’re looking for..."
          style={styles.input}
          disabled={loading}
        />
        <button
          onClick={handleSearch}
          style={{
            ...styles.button,
            opacity: loading ? 0.7 : 1,
            cursor: loading ? "not-allowed" : "pointer",
          }}
          disabled={loading}
        >
          {loading ? "Searching..." : "Search"} {/* 🔹 Dynamic label */}
        </button>
      </div>

      {loading && <p style={styles.loadingText}>Fetching papers...</p>} {/* 🔹 Optional loading text */}

      <div style={styles.resultsContainer}>
        {results.map((paper, i) => (
          <div key={i} style={styles.card}>
            <h3 style={styles.paperTitle}>{paper.title}</h3>
            <p style={styles.authors}>
              {paper.authors?.length ? paper.authors.join(", ") : "Unknown authors"}{" "}
              <span style={styles.year}>({paper.year || "n/a"})</span>
            </p>

            {expanded[i] ? (
              <>
                <p style={styles.abstract}>{paper.abstract || "No abstract available."}</p>
                <button style={styles.toggleBtn} onClick={() => toggleAbstract(i)}>
                  Hide abstract
                </button>
              </>
            ) : (
              <button style={styles.toggleBtn} onClick={() => toggleAbstract(i)}>
                Show abstract
              </button>
            )}
            <br />
            <a href={paper.url} target="_blank" rel="noreferrer" style={styles.link}>
              View on Semantic Scholar ↗
            </a>
          </div>
        ))}
      </div>
    </div>
  );
}

// 🌿 Style system
const styles = {
  container: {
    fontFamily: "Inter, Arial, sans-serif",
    backgroundColor: "#e6f2ef",
    minHeight: "100vh",
    padding: "2rem",
    color: "#0b3b36",
  },
  header: { textAlign: "center", marginBottom: "2rem" },
  title: { fontSize: "2.5rem", color: "#156f63", margin: 0 },
  subtitle: { fontSize: "1rem", color: "#285c54", marginTop: "0.5rem" },
  searchBox: {
    display: "flex",
    justifyContent: "center",
    marginBottom: "2rem",
  },
  input: {
    width: "400px",
    padding: "0.8rem",
    borderRadius: "10px",
    border: "1px solid #a4ccc1",
    outline: "none",
    marginRight: "10px",
    fontSize: "1rem",
  },
  button: {
    backgroundColor: "#1c776a",
    color: "white",
    padding: "0.8rem 1.4rem",
    border: "none",
    borderRadius: "10px",
    fontWeight: "bold",
  },
  loadingText: {
    textAlign: "center",
    fontStyle: "italic",
    color: "#285c54",
    marginBottom: "1rem",
  },
  resultsContainer: {
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
  },
  card: {
    backgroundColor: "#f8fbfa",
    border: "1px solid #c4e3da",
    borderRadius: "14px",
    boxShadow: "0 4px 12px rgba(0, 0, 0, 0.05)",
    padding: "1.5rem",
    marginBottom: "1.5rem",
    width: "80%",
    maxWidth: "800px",
  },
  paperTitle: { color: "#10463e", marginBottom: "0.3rem" },
  authors: { fontStyle: "italic", color: "#2f5f59", marginBottom: "0.6rem" },
  year: { color: "#5e8b83" },
  abstract: { color: "#23433f", lineHeight: 1.5 },
  toggleBtn: {
    marginTop: "0.5rem",
    backgroundColor: "transparent",
    border: "none",
    color: "#1c776a",
    cursor: "pointer",
    fontWeight: "bold",
  },
  link: {
    display: "inline-block",
    marginTop: "1rem",
    color: "#007b7f",
    textDecoration: "none",
    fontWeight: "bold",
  },
};

export default App;
