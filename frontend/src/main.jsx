import React, { useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  AlertTriangle,
  Boxes,
  CheckCircle2,
  Copy,
  Database,
  FileText,
  LoaderCircle,
  RefreshCw,
  Search,
  ShieldCheck,
  Table2,
  Upload,
  Wrench,
  X,
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

function Metric({ label, value, active = false, onClick, title }) {
  return (
    <button
      type="button"
      className={`metric ${active ? "active" : ""}`}
      onClick={onClick}
      title={title}
      aria-pressed={active}
      aria-controls="parsed-table-results"
    >
      <span>{label}</span>
      <strong>{value ?? "-"}</strong>
    </button>
  );
}

const TABLE_FILTERS = {
  pages: {
    label: "All pages",
    matches: () => true,
  },
  tables: {
    label: "All tables",
    matches: () => true,
  },
  rawRows: {
    label: "Tables with raw rows",
    matches: (table) => (table.raw_rows?.length ?? table.raw_parsed_table?.rows?.length ?? 0) > 0,
  },
  repairs: {
    label: "Tables with repairs",
    matches: (table) => (table.repairs?.length ?? 0) > 0,
  },
  review: {
    label: "Tables needing review",
    matches: (table) => table.quality?.status === "NEEDS_REVIEW",
  },
  warnings: {
    label: "Tables with warnings",
    matches: (table) => (
      (table.parsed_table?.parse_warnings?.length ?? 0)
      + (table.quality?.remaining_issues?.length ?? 0)
    ) > 0,
  },
};

function StatusBadge({ status }) {
  const ready = status === "READY";
  return (
    <span className={`status ${ready ? "ready" : "pending"}`}>
      {ready ? <CheckCircle2 size={13} /> : <LoaderCircle size={13} />}
      {status}
    </span>
  );
}

function TableGrid({ table }) {
  return (
    <div className="table-scroll" role="tabpanel">
      <table>
        <thead>
          <tr>
            {table.columns.map((column) => (
              <th key={column.key}>{column.label}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {table.rows.map((row) => (
            <tr key={row.row_index} className={row.is_total ? "total" : ""}>
              {row.raw_cells.map((cell, index) => (
                <td key={`${row.row_index}-${index}`} className={cell == null ? "missing-cell" : ""}>
                  {cell ?? "NULL"}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function TablePanel({ table }) {
  const [tab, setTab] = useState("raw");
  const parsed = table.parsed_table;
  const raw = table.raw_parsed_table || parsed;
  const quality = table.quality || { status: "PASS", confidence: 1, initial_issues: [], remaining_issues: [] };
  const tabs = [
    ["raw", "Raw extraction"],
    ["canonical", "Canonical table"],
    ["markdown", "Original Markdown"],
    ["semantic", "Semantic rows"],
    ["quality", "Quality & repairs"],
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
          <span className={`quality-chip ${quality.status.toLowerCase()}`}>{quality.status.replace("_", " ")}</span>
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

      {tab === "raw" && <TableGrid table={raw} />}

      {tab === "canonical" && <TableGrid table={parsed} />}

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

      {tab === "quality" && (
        <div className="quality-panel" role="tabpanel">
          <div className="quality-summary">
            <ShieldCheck size={18} />
            <div>
              <strong>{quality.status.replace("_", " ")}</strong>
              <span>{Math.round(quality.confidence * 100)}% structural confidence</span>
            </div>
          </div>

          <section>
            <h3>Initial issues</h3>
            {quality.initial_issues.length === 0 ? <p className="quiet">No structural issues detected.</p> : (
              <div className="issue-list">
                {quality.initial_issues.map((issue, index) => (
                  <div className="issue-row" key={`${issue.code}-${index}`}>
                    <code>{issue.code}</code>
                    <span>{issue.message}</span>
                    <small>{issue.severity}{issue.row_index ? ` · row ${issue.row_index}` : ""}</small>
                  </div>
                ))}
              </div>
            )}
          </section>

          <section>
            <h3>Applied repairs</h3>
            {table.repairs.length === 0 ? <p className="quiet">No repairs applied.</p> : (
              <div className="repair-list">
                {table.repairs.map((repair, index) => (
                  <div className="repair-row" key={`${repair.row_index}-${repair.column_index}-${index}`}>
                    <Wrench size={16} />
                    <div>
                      <strong>{repair.operation.replaceAll("_", " ")} · row {repair.row_index}, column {repair.column_index + 1}</strong>
                      <p>{repair.original_value ?? "NULL"} → {repair.repaired_value}</p>
                      <small>{repair.source}{repair.model ? ` · ${repair.model}` : ""} · {Math.round(repair.confidence * 100)}% · {repair.source_token_ids.length} source tokens</small>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </section>

          <section>
            <h3>Remaining issues</h3>
            {quality.remaining_issues.length === 0 ? <p className="quiet">No unresolved structural issues.</p> : (
              <div className="issue-list">
                {quality.remaining_issues.map((issue, index) => (
                  <div className="issue-row" key={`${issue.code}-${index}`}>
                    <code>{issue.code}</code><span>{issue.message}</span><small>{issue.severity}</small>
                  </div>
                ))}
              </div>
            )}
          </section>
        </div>
      )}
    </article>
  );
}

function ChunkExplorer({ chunks, selectedChunkId, onSelect, detail, loading }) {
  const [typeFilter, setTypeFilter] = useState("all");
  const [pageFilter, setPageFilter] = useState("all");
  const [query, setQuery] = useState("");
  const [copied, setCopied] = useState(false);
  const types = useMemo(
    () => [...new Set(chunks.map((chunk) => chunk.chunk_type))].sort(),
    [chunks],
  );
  const pages = useMemo(
    () => [...new Set(chunks.map((chunk) => chunk.page_number))].sort((a, b) => a - b),
    [chunks],
  );
  const filteredChunks = useMemo(() => {
    const normalizedQuery = query.trim().toLowerCase();
    return chunks.filter((chunk) => (
      (typeFilter === "all" || chunk.chunk_type === typeFilter)
      && (pageFilter === "all" || chunk.page_number === Number(pageFilter))
      && (
        !normalizedQuery
        || chunk.content_preview.toLowerCase().includes(normalizedQuery)
        || (chunk.comparison_key || "").toLowerCase().includes(normalizedQuery)
        || chunk.representation.toLowerCase().includes(normalizedQuery)
      )
    ));
  }, [chunks, pageFilter, query, typeFilter]);

  async function copyContent() {
    if (!detail) return;
    await navigator.clipboard.writeText(detail.content);
    setCopied(true);
    window.setTimeout(() => setCopied(false), 1400);
  }

  return (
    <section className="chunk-section" aria-label="Chunk explorer">
      <div className="chunk-toolbar">
        <div>
          <strong>Chunks</strong>
          <span>{filteredChunks.length} of {chunks.length}</span>
        </div>
        <div className="chunk-filters">
          <label className="search-field">
            <Search size={15} />
            <input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search content or key" />
          </label>
          <label>
            Type
            <select value={typeFilter} onChange={(event) => setTypeFilter(event.target.value)}>
              <option value="all">All types</option>
              {types.map((type) => <option key={type} value={type}>{type}</option>)}
            </select>
          </label>
          <label>
            Page
            <select value={pageFilter} onChange={(event) => setPageFilter(event.target.value)}>
              <option value="all">All pages</option>
              {pages.map((page) => <option key={page} value={page}>{page}</option>)}
            </select>
          </label>
        </div>
      </div>

      <div className="chunk-explorer">
        <div className="chunk-list" role="listbox" aria-label="Document chunks">
          {filteredChunks.map((chunk) => (
            <button
              type="button"
              role="option"
              aria-selected={selectedChunkId === chunk.id}
              key={chunk.id}
              className={`chunk-item ${selectedChunkId === chunk.id ? "selected" : ""}`}
              onClick={() => onSelect(chunk.id)}
            >
              <span className={`chunk-type type-${chunk.chunk_type.toLowerCase()}`}>{chunk.chunk_type}</span>
              <small>Page {chunk.page_number} · {chunk.representation}</small>
              <strong>{chunk.comparison_key || chunk.id}</strong>
              <p>{chunk.content_preview || "Empty content"}</p>
            </button>
          ))}
          {filteredChunks.length === 0 && <p className="chunk-list-empty">No chunks match these filters.</p>}
        </div>

        <article className="chunk-detail">
          {loading ? (
            <div className="chunk-detail-empty"><LoaderCircle className="spin" size={24} /> Loading chunk</div>
          ) : !detail ? (
            <div className="chunk-detail-empty"><Boxes size={28} /> Select a chunk to inspect it.</div>
          ) : (
            <>
              <header className="chunk-detail-header">
                <div>
                  <div className="eyebrow">Page {detail.page_number} · {detail.chunk_type}</div>
                  <h2>{detail.comparison_key || detail.id}</h2>
                </div>
                <button type="button" className="copy-button" onClick={copyContent} title="Copy chunk content">
                  <Copy size={15} /> {copied ? "Copied" : "Copy content"}
                </button>
              </header>

              <dl className="chunk-properties">
                <div><dt>Chunk ID</dt><dd>{detail.id}</dd></div>
                <div><dt>Document ID</dt><dd>{detail.document_id}</dd></div>
                <div><dt>Representation</dt><dd>{detail.representation}</dd></div>
                <div><dt>Embedding</dt><dd>{detail.embedding_dimensions == null ? "Not generated" : `${detail.embedding_dimensions} dimensions`}</dd></div>
                <div><dt>Created</dt><dd>{new Date(detail.created_at).toLocaleString()}</dd></div>
              </dl>

              <section className="chunk-content-section">
                <h3>Content</h3>
                <pre>{detail.content}</pre>
              </section>
              {detail.raw_content != null && (
                <section className="chunk-content-section">
                  <h3>Raw content</h3>
                  <pre>{detail.raw_content}</pre>
                </section>
              )}
              {detail.semantic_content != null && (
                <section className="chunk-content-section">
                  <h3>Semantic content</h3>
                  <pre>{detail.semantic_content}</pre>
                </section>
              )}
              <section className="chunk-content-section">
                <h3>Metadata</h3>
                <pre>{JSON.stringify(detail.metadata, null, 2)}</pre>
              </section>
            </>
          )}
        </article>
      </div>
    </section>
  );
}

function App() {
  const inputRef = useRef(null);
  const [documents, setDocuments] = useState([]);
  const [selectedId, setSelectedId] = useState(null);
  const [tables, setTables] = useState([]);
  const [chunks, setChunks] = useState([]);
  const [selectedChunkId, setSelectedChunkId] = useState(null);
  const [selectedChunk, setSelectedChunk] = useState(null);
  const [chunkLoading, setChunkLoading] = useState(false);
  const [viewMode, setViewMode] = useState("tables");
  const [tableFilter, setTableFilter] = useState("pages");
  const [pageFilter, setPageFilter] = useState("all");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  const selected = documents.find((document) => document.id === selectedId);
  const statusFilteredTables = useMemo(
    () => tables.filter(TABLE_FILTERS[tableFilter].matches),
    [tables, tableFilter],
  );
  const pages = useMemo(
    () => [...new Set(statusFilteredTables.map((table) => table.parsed_table.page_number))].sort((a, b) => a - b),
    [statusFilteredTables],
  );
  const filteredTables = pageFilter === "all"
    ? statusFilteredTables
    : statusFilteredTables.filter((table) => table.parsed_table.page_number === Number(pageFilter));

  function applyTableFilter(filter) {
    setViewMode("tables");
    setTableFilter(filter);
    setPageFilter("all");
  }

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
    setTableFilter("pages");
    setPageFilter("all");
  }

  async function loadChunks(documentId) {
    if (!documentId) {
      setChunks([]);
      setSelectedChunkId(null);
      setSelectedChunk(null);
      return;
    }
    const result = await api(`/api/documents/${documentId}/chunks`);
    setChunks(result);
    setSelectedChunkId((current) => (
      result.some((chunk) => chunk.id === current) ? current : result[0]?.id || null
    ));
    setSelectedChunk(null);
  }

  useEffect(() => {
    loadDocuments().catch((err) => setError(err.message));
  }, []);

  useEffect(() => {
    Promise.all([loadTables(selectedId), loadChunks(selectedId)]).catch((err) => setError(err.message));
  }, [selectedId]);

  useEffect(() => {
    if (viewMode !== "chunks" || !selectedId || !selectedChunkId) return undefined;
    let active = true;
    setSelectedChunk(null);
    setChunkLoading(true);
    api(`/api/documents/${selectedId}/chunks/${selectedChunkId}`)
      .then((result) => { if (active) setSelectedChunk(result); })
      .catch((err) => { if (active) setError(err.message); })
      .finally(() => { if (active) setChunkLoading(false); });
    return () => { active = false; };
  }, [selectedChunkId, selectedId, viewMode]);

  async function uploadFile(file) {
    if (!file) return;
    setBusy(true);
    setError("");
    const body = new FormData();
    body.append("file", file);
    try {
      const document = await api("/api/documents", { method: "POST", body });
      await loadDocuments(document.id);
      await Promise.all([loadTables(document.id), loadChunks(document.id)]);
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
                  <Metric label="Pages" value={selected.page_count} active={viewMode === "tables" && tableFilter === "pages"} onClick={() => applyTableFilter("pages")} title="Show tables from all parsed pages" />
                  <Metric label="Tables" value={selected.metadata.tables} active={viewMode === "tables" && tableFilter === "tables"} onClick={() => applyTableFilter("tables")} title="Show every detected table" />
                  <Metric label="Raw rows" value={selected.metadata.raw_rows} active={viewMode === "tables" && tableFilter === "rawRows"} onClick={() => applyTableFilter("rawRows")} title="Show tables containing extracted rows" />
                  <Metric label="Repairs" value={selected.metadata.table_repairs ?? 0} active={viewMode === "tables" && tableFilter === "repairs"} onClick={() => applyTableFilter("repairs")} title="Show pages where repairs were applied" />
                  <Metric label="Review" value={selected.metadata.needs_review_tables ?? 0} active={viewMode === "tables" && tableFilter === "review"} onClick={() => applyTableFilter("review")} title="Show pages that still need review" />
                  <Metric label="Warnings" value={selected.metadata.parse_warnings} active={viewMode === "tables" && tableFilter === "warnings"} onClick={() => applyTableFilter("warnings")} title="Show pages with warnings or unresolved issues" />
                </div>
              </section>

              <div className="view-switcher" role="tablist" aria-label="Document debug views">
                <button type="button" role="tab" aria-selected={viewMode === "tables"} className={viewMode === "tables" ? "active" : ""} onClick={() => setViewMode("tables")}>
                  <Table2 size={16} /> Tables <span>{tables.length}</span>
                </button>
                <button type="button" role="tab" aria-selected={viewMode === "chunks"} className={viewMode === "chunks" ? "active" : ""} onClick={() => setViewMode("chunks")}>
                  <Boxes size={16} /> Chunks <span>{chunks.length}</span>
                </button>
              </div>

              {viewMode === "tables" ? <><div className="table-toolbar">
                <div>
                  <strong>Parsed tables</strong>
                  <span>
                    {TABLE_FILTERS[tableFilter].label} · {filteredTables.length} {filteredTables.length === 1 ? "table" : "tables"}
                    {pageFilter === "all" ? ` on ${pages.length} ${pages.length === 1 ? "page" : "pages"}` : ` on page ${pageFilter}`}
                  </span>
                </div>
                <div className="toolbar-actions">
                  <label>
                    Page
                    <select value={pageFilter} onChange={(event) => setPageFilter(event.target.value)}>
                      <option value="all">All matching pages</option>
                      {pages.map((page) => <option key={page} value={page}>{page}</option>)}
                    </select>
                  </label>
                  {(tableFilter !== "pages" || pageFilter !== "all") && (
                    <button className="clear-filter" type="button" onClick={() => applyTableFilter("pages")}>
                      <X size={15} /> Clear filter
                    </button>
                  )}
                </div>
              </div>

              {!["pages", "tables", "rawRows"].includes(tableFilter) && pages.length > 0 && (
                <nav className="matching-pages" aria-label={`${TABLE_FILTERS[tableFilter].label} page filters`}>
                  <span>Matching pages</span>
                  <button type="button" className={pageFilter === "all" ? "active" : ""} onClick={() => setPageFilter("all")}>All</button>
                  {pages.map((page) => (
                    <button
                      type="button"
                      key={page}
                      className={Number(pageFilter) === page ? "active" : ""}
                      onClick={() => setPageFilter(String(page))}
                    >
                      {page}
                    </button>
                  ))}
                </nav>
              )}

              <div className="table-list" id="parsed-table-results" aria-live="polite">
                {filteredTables.map((table) => (
                  <TablePanel key={table.parsed_table.table_id} table={table} />
                ))}
                {filteredTables.length === 0 && (
                  <div className="filter-empty">
                    <CheckCircle2 size={28} />
                    <strong>No matching tables</strong>
                    <span>This document has no tables matching {TABLE_FILTERS[tableFilter].label.toLowerCase()}.</span>
                    <button type="button" onClick={() => applyTableFilter("pages")}>Show all parsed pages</button>
                  </div>
                )}
              </div>
              </> : (
                <ChunkExplorer
                  key={selectedId}
                  chunks={chunks}
                  selectedChunkId={selectedChunkId}
                  onSelect={setSelectedChunkId}
                  detail={selectedChunk}
                  loading={chunkLoading}
                />
              )}
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
