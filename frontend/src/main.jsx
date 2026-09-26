import React, { useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  AlertTriangle,
  CheckCircle2,
  Database,
  FileText,
  LoaderCircle,
  RefreshCw,
  Table2,
  Upload,
} from "lucide-react";
import "./styles.css";

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

async function api(path, options) {
  const response = await fetch(`${API_URL}${path}`, options);
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.detail || `Request failed (${response.status})`);
  }
  return response.json();
}

function Metric({ label, value }) {
  return (
    <div className="metric">
      <span>{label}</span>
      <strong>{value ?? "-"}</strong>
    </div>
  );
}

function StatusBadge({ status }) {
  const ready = status === "READY";
  return (
    <span className={`status ${ready ? "ready" : "pending"}`}>
      {ready ? <CheckCircle2 size={13} /> : <LoaderCircle size={13} />}
      {status}
    </span>
  );
}

function TablePanel({ table }) {
  const [tab, setTab] = useState("raw");
  const parsed = table.parsed_table;
  const tabs = [
    ["raw", "Raw extraction"],
    ["markdown", "Original Markdown"],
    ["semantic", "Semantic rows"],
  ];

  return (
    <article className="table-panel">
      <div className="table-heading">
        <div>
          <div className="eyebrow">
            Page {parsed.page_number} · {parsed.source_kind} · {parsed.extraction_method}
          </div>
          <h2>{parsed.title}</h2>
          {parsed.subtitle && <p>{parsed.subtitle}</p>}
        </div>
        <div className="table-stats" aria-label="Table dimensions">
          <span>{parsed.rows.length} rows</span>
          <span>{parsed.columns.length} columns</span>
        </div>
      </div>

      {parsed.parse_warnings.length > 0 && (
        <div className="warning">
          <AlertTriangle size={17} />
          <div>
            {parsed.parse_warnings.map((warning) => (
              <p key={warning}>{warning}</p>
            ))}
          </div>
        </div>
      )}

      <div className="tabs" role="tablist" aria-label={`${parsed.title} representations`}>
        {tabs.map(([id, label]) => (
          <button
            key={id}
            className={tab === id ? "active" : ""}
            onClick={() => setTab(id)}
            role="tab"
            aria-selected={tab === id}
          >
            {label}
          </button>
        ))}
      </div>

      {tab === "raw" && (
        <div className="table-scroll" role="tabpanel">
          <table>
            <thead>
              <tr>
                {parsed.columns.map((column) => (
                  <th key={column.key}>{column.label}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {parsed.rows.map((row) => (
                <tr key={row.row_index} className={row.is_total ? "total" : ""}>
                  {row.raw_cells.map((cell, index) => (
                    <td key={`${row.row_index}-${index}`}>{cell ?? ""}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {tab === "markdown" && (
        <pre className="code-block" role="tabpanel">{table.markdown}</pre>
      )}

      {tab === "semantic" && (
        <div className="semantic-list" role="tabpanel">
          {table.semantic_rows.map((row) => (
            <div className="semantic-row" key={row.comparison_key}>
              <div className="semantic-label">
                <span>{row.row_label}</span>
                <code>{row.comparison_key}</code>
              </div>
              <pre>{row.content}</pre>
            </div>
          ))}
        </div>
      )}
    </article>
  );
}

function App() {
  const inputRef = useRef(null);
  const [documents, setDocuments] = useState([]);
  const [selectedId, setSelectedId] = useState(null);
  const [tables, setTables] = useState([]);
  const [pageFilter, setPageFilter] = useState("all");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  const selected = documents.find((document) => document.id === selectedId);
  const pages = useMemo(
    () => [...new Set(tables.map((table) => table.parsed_table.page_number))].sort((a, b) => a - b),
    [tables],
  );
  const filteredTables = pageFilter === "all"
    ? tables
    : tables.filter((table) => table.parsed_table.page_number === Number(pageFilter));

  async function loadDocuments(preferredId) {
    const result = await api("/api/documents");
    setDocuments(result);
    const nextId = preferredId || selectedId || result[0]?.id || null;
    setSelectedId(nextId);
    return nextId;
  }

  async function loadTables(documentId) {
    if (!documentId) {
      setTables([]);
      return;
    }
    setTables(await api(`/api/documents/${documentId}/tables`));
    setPageFilter("all");
  }

  useEffect(() => {
    loadDocuments().catch((err) => setError(err.message));
  }, []);

  useEffect(() => {
    loadTables(selectedId).catch((err) => setError(err.message));
  }, [selectedId]);

  async function uploadFile(file) {
    if (!file) return;
    setBusy(true);
    setError("");
    const body = new FormData();
    body.append("file", file);
    try {
      const document = await api("/api/documents", { method: "POST", body });
      await loadDocuments(document.id);
      await loadTables(document.id);
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
      if (inputRef.current) inputRef.current.value = "";
    }
  }

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand">
          <div className="brand-mark"><Table2 size={21} /></div>
          <div>
            <h1>Semantic Table RAG POC</h1>
            <p>Ingestion and representation debugger</p>
          </div>
        </div>
        <div className="topbar-actions">
          <button
            className="icon-button"
            title="Refresh documents"
            aria-label="Refresh documents"
            onClick={() => loadDocuments().catch((err) => setError(err.message))}
          >
            <RefreshCw size={18} />
          </button>
          <input
            ref={inputRef}
            type="file"
            accept="application/pdf,.pdf"
            hidden
            onChange={(event) => uploadFile(event.target.files?.[0])}
          />
          <button className="primary-button" onClick={() => inputRef.current?.click()} disabled={busy}>
            {busy ? <LoaderCircle className="spin" size={17} /> : <Upload size={17} />}
            {busy ? "Parsing PDF" : "Upload PDF"}
          </button>
        </div>
      </header>

      {error && (
        <div className="error-banner">
          <AlertTriangle size={17} />
          <span>{error}</span>
        </div>
      )}

      <div className="workspace">
        <aside className="sidebar">
          <div className="section-label"><Database size={15} /> Documents</div>
          <div className="document-list">
            {documents.map((document) => (
              <button
                key={document.id}
                className={`document-item ${selectedId === document.id ? "selected" : ""}`}
                onClick={() => setSelectedId(document.id)}
              >
                <FileText size={17} />
                <span>
                  <strong>{document.filename}</strong>
                  <small>{document.page_count ?? 0} pages · {document.metadata.tables ?? 0} tables</small>
                </span>
              </button>
            ))}
            {documents.length === 0 && <p className="empty-sidebar">No PDFs uploaded</p>}
          </div>
        </aside>

        <main className="main-content">
          {!selected ? (
            <div className="empty-state">
              <Table2 size={34} />
              <h2>No parsed document</h2>
              <p>Upload the AIA analyst presentation to inspect its extracted tables.</p>
              <button className="primary-button" onClick={() => inputRef.current?.click()}>
                <Upload size={17} /> Upload PDF
              </button>
            </div>
          ) : (
            <>
              <section className="document-header">
                <div>
                  <div className="eyebrow">Parsed document</div>
                  <h2>{selected.filename}</h2>
                  <StatusBadge status={selected.status} />
                </div>
                <div className="metrics">
                  <Metric label="Pages" value={selected.page_count} />
                  <Metric label="Tables" value={selected.metadata.tables} />
                  <Metric label="Raw rows" value={selected.metadata.raw_rows} />
                  <Metric label="Warnings" value={selected.metadata.parse_warnings} />
                </div>
              </section>

              <div className="table-toolbar">
                <div>
                  <strong>Parsed tables</strong>
                  <span>{filteredTables.length} shown</span>
                </div>
                <label>
                  Page
                  <select value={pageFilter} onChange={(event) => setPageFilter(event.target.value)}>
                    <option value="all">All pages</option>
                    {pages.map((page) => <option key={page} value={page}>{page}</option>)}
                  </select>
                </label>
              </div>

              <div className="table-list">
                {filteredTables.map((table) => (
                  <TablePanel key={table.parsed_table.table_id} table={table} />
                ))}
              </div>
            </>
          )}
        </main>
      </div>
    </div>
  );
}

createRoot(document.getElementById("root")).render(
  <React.StrictMode><App /></React.StrictMode>,
);

