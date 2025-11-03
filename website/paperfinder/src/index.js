import React, { useEffect } from "react";
import ReactDOM from "react-dom/client";
import "./index.css";
import App from "./App";

// Load MathJax v3 once at bootstrap
function Bootstrapper() {
  useEffect(() => {
    if (!window.MathJax) {
      // Configure MathJax for \( ... \) inline and \[ ... \] display math
      window.MathJax = {
        tex: {
          inlineMath: [["\\(", "\\)"], ["$", "$"]],
          displayMath: [["\\[", "\\]"], ["$$", "$$"]],
          processEscapes: true,
        },
        options: {
          skipHtmlTags: ["script", "noscript", "style", "textarea", "pre", "code"],
        },
        svg: { fontCache: "global" },
      };

      const script = document.createElement("script");
      script.id = "mathjax-script";
      script.type = "text/javascript";
      script.src = "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js";
      script.async = true;
      document.head.appendChild(script);
    }
  }, []);

  return (
    <React.StrictMode>
      <App />
    </React.StrictMode>
  );
}

const root = ReactDOM.createRoot(document.getElementById("root"));
root.render(<Bootstrapper />);
