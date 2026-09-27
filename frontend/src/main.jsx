import React, { useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  AlertTriangle,
  Boxes,
  CheckCircle2,
  Clock3,
  Copy,
  Cpu,
  Database,
  FileText,
  GitMerge,
  Library,
  LoaderCircle,
  Plus,
  RefreshCw,
  Search,
  ScanSearch,
  ShieldCheck,
  Table2,
  Upload,
  Wrench,
  X,
} from "lucide-react";
import "./styles.css";

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

function formatDuration(seconds) {
  if (seconds == null) return null;
  if (seconds < 60) return `${seconds.toFixed(1)}s`;
  return `${Math.floor(seconds / 60)}m ${Math.round(seconds % 60)}s`;
}

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
  const failed = status === "FAILED";
  return (
    <span className={`status ${ready ? "ready" : failed ? "failed" : "pending"}`}>
      {ready ? <CheckCircle2 size={13} /> : failed ? <AlertTriangle size={13} /> : <LoaderCircle className="spin" size={13} />}
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

const RETRIEVAL_STAGES = [
  ["vector_results", "Vector"],
  ["lexical_results", "Lexical"],
  ["rrf_results", "RRF merged"],
  ["reranked_results", "Reranked"],
];

function Score({ label, value }) {
  return <span><small>{label}</small>{value == null ? "-" : value.toFixed(6)}</span>;
}

function RetrievalHit({ hit }) {
  return (
    <article className="retrieval-hit">
      <header>
        <div className="retrieval-ranks">
          <strong>#{hit.rank}</strong>
          <span>Final {hit.final_rank == null ? "-" : `#${hit.final_rank}`}</span>
        </div>
        <div className="retrieval-identity">
          <span className={`chunk-type type-${hit.chunk_type.toLowerCase()}`}>{hit.chunk_type}</span>
          <strong>{hit.comparison_key || hit.chunk_id}</strong>
          <small>{hit.file} · Page {hit.page} · {hit.representation}</small>
        </div>
      </header>
      <dl className="retrieval-fields">
        <div><dt>Table title</dt><dd>{hit.table_title || "-"}</dd></div>
        <div><dt>Row label</dt><dd>{hit.row_label || "-"}</dd></div>
      </dl>
      <div className="retrieval-scores">
        <Score label="Vector" value={hit.vector_score} />
        <Score label="Lexical" value={hit.lexical_score} />
        <Score label="RRF" value={hit.rrf_score} />
        <Score label="Rerank" value={hit.rerank_score} />
      </div>
      <div className="retrieval-content-grid">
        <section><h4>Raw content</h4><pre>{hit.raw_content || hit.content}</pre></section>
        <section><h4>Semantic content</h4><pre>{hit.semantic_content || hit.content}</pre></section>
        {hit.parent_content && <section className="parent-context"><h4>Parent context</h4><pre>{hit.parent_content}</pre></section>}
      </div>
    </article>
  );
}

function RetrievalDebugger({ documentId, knowledgeBaseId, knowledgeBaseOnly = false, onError }) {
  const [query, setQuery] = useState("2025年底AIA美国公司债有多少，占公司债组合多少？");
  const [mode, setMode] = useState("PRODUCTION");
  const [scope, setScope] = useState("KNOWLEDGE_BASE");
  const [company, setCompany] = useState("");
  const [fiscalYear, setFiscalYear] = useState("");
  const [stage, setStage] = useState("reranked_results");
  const [result, setResult] = useState(null);
  const [busy, setBusy] = useState(false);
  const [indexing, setIndexing] = useState(false);
  const [indexStatus, setIndexStatus] = useState(null);

  async function buildIndex() {
    setIndexing(true);
    onError("");
    try {
      const path = scope === "KNOWLEDGE_BASE"
        ? `/api/knowledge-bases/${knowledgeBaseId}/index`
        : `/api/retrieval/documents/${documentId}/index-async`;
      setIndexStatus(await api(path, {
        method: "POST",
      }));
    } catch (err) {
      onError(err.message);
    } finally {
      setIndexing(false);
    }
  }

  async function runSearch(event) {
    event.preventDefault();
    if (!query.trim()) return;
    setBusy(true);
    onError("");
    try {
      const retrievalScope = scope === "KNOWLEDGE_BASE"
        ? {
            knowledge_base_id: knowledgeBaseId,
            ...(company.trim() ? { company: company.trim() } : {}),
            ...(fiscalYear ? { fiscal_year: Number(fiscalYear) } : {}),
          }
        : { document_id: documentId };
      setResult(await api("/api/retrieval/search", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ...retrievalScope, query: query.trim(), mode }),
      }));
      setStage("reranked_results");
    } catch (err) {
      onError(err.message);
    } finally {
      setBusy(false);
    }
  }

  const hits = result?.[stage] || [];
  return (
    <section className="retrieval-debugger">
      <form className={`retrieval-query ${knowledgeBaseOnly ? "knowledge-base-query" : ""}`} onSubmit={runSearch}>
        <div className="retrieval-query-main">
          <label htmlFor="retrieval-query">Query</label>
          <div className="retrieval-query-input">
            <Search size={17} />
            <input id="retrieval-query" value={query} onChange={(event) => setQuery(event.target.value)} />
          </div>
        </div>
        <div className="mode-control" aria-label="Retrieval mode">
          {[
            ["BASELINE", "Baseline"],
            ["SEMANTIC", "Semantic"],
            ["PRODUCTION", "Production"],
          ].map(([value, label]) => (
            <button type="button" key={value} className={mode === value ? "active" : ""} onClick={() => setMode(value)}>{label}</button>
          ))}
        </div>
        {!knowledgeBaseOnly && (
          <div className="mode-control" aria-label="Retrieval scope">
            {[
              ["DOCUMENT", "Current file"],
              ["KNOWLEDGE_BASE", "Knowledge base"],
            ].map(([value, label]) => (
              <button type="button" key={value} className={scope === value ? "active" : ""} onClick={() => setScope(value)}>{label}</button>
            ))}
          </div>
        )}
        <button type="button" className="secondary-button" onClick={buildIndex} disabled={indexing}>
          {indexing ? <LoaderCircle className="spin" size={16} /> : <Database size={16} />}
          {indexing ? "Indexing" : "Build index"}
        </button>
        <button type="submit" className="primary-button" disabled={busy}>
          {busy ? <LoaderCircle className="spin" size={16} /> : <Search size={16} />}
          {busy ? "Retrieving" : "Run retrieval"}
        </button>
      </form>
      {scope === "KNOWLEDGE_BASE" && (
        <div className="retrieval-metadata-filters">
          <label>Company<input value={company} onChange={(event) => setCompany(event.target.value)} placeholder="All companies" /></label>
          <label>Fiscal year<input type="number" min="1900" max="2200" value={fiscalYear} onChange={(event) => setFiscalYear(event.target.value)} placeholder="All years" /></label>
        </div>
      )}
      {indexStatus && (
        <div className="index-status">
          {indexStatus.queued_jobs != null
            ? `${indexStatus.queued_jobs} indexing jobs queued · ${indexStatus.skipped_active_jobs} already active`
            : indexStatus.stage
              ? `Index job ${indexStatus.status.toLowerCase()} · ${indexStatus.stage}`
            : `${indexStatus.indexed_chunks} indexed · ${indexStatus.skipped_chunks} already present · ${indexStatus.model} / ${indexStatus.dimensions}d`}
        </div>
      )}
      {!result ? (
        <div className="retrieval-empty"><GitMerge size={28} /><strong>No retrieval run yet</strong></div>
      ) : (
        <>
          <div className="retrieval-run-meta">
            <span><strong>Query</strong> {result.original_query}</span>
            <span><strong>Mode</strong> {result.retrieval_mode}</span>
            <span><strong>Documents</strong> {result.document_ids.length}</span>
            <span><strong>Models</strong> {result.embedding_model} · {result.rerank_model}</span>
          </div>
          <div className="stage-tabs" role="tablist" aria-label="Retrieval pipeline stages">
            {RETRIEVAL_STAGES.map(([id, label]) => (
              <button type="button" role="tab" aria-selected={stage === id} className={stage === id ? "active" : ""} onClick={() => setStage(id)} key={id}>
                {label}<span>{result[id].length}</span>
              </button>
            ))}
          </div>
          <div className="retrieval-results">
            {hits.map((hit) => <RetrievalHit hit={hit} key={hit.chunk_id} />)}
            {hits.length === 0 && <div className="retrieval-empty"><Search size={24} /><strong>No hits at this stage</strong></div>}
          </div>
        </>
      )}
    </section>
  );
}

function UploadDialog({ file, busy, onCancel, onSubmit }) {
  const yearMatch = file.name.match(/\b(20\d{2})\b/);
  const [company, setCompany] = useState(file.name.match(/^(.+?)\s+(?:20\d{2}|Annual|Interim)/i)?.[1] || "");
  const [fiscalYear, setFiscalYear] = useState(yearMatch?.[1] || "");
  const [documentType, setDocumentType] = useState(
    /annual results/i.test(file.name) ? "annual_results" : /interim/i.test(file.name) ? "interim_results" : "",
  );
  const [language, setLanguage] = useState(/\bEN\b/i.test(file.name) ? "en" : "");

  function submit(event) {
    event.preventDefault();
    onSubmit(file, {
      company: company.trim(),
      fiscal_year: fiscalYear,
      document_type: documentType,
      language: language.trim(),
    });
  }

  return (
    <div className="dialog-backdrop" role="presentation">
      <form className="upload-dialog" onSubmit={submit} role="dialog" aria-modal="true" aria-labelledby="upload-title">
        <header>
          <div><div className="eyebrow">Queue finance document</div><h2 id="upload-title">{file.name}</h2></div>
          <button type="button" className="icon-button" onClick={onCancel} aria-label="Close upload dialog"><X size={17} /></button>
        </header>
        <div className="upload-metadata-grid">
          <label>Company<input value={company} onChange={(event) => setCompany(event.target.value)} placeholder="e.g. AIA Group" /></label>
          <label>Fiscal year<input type="number" min="1900" max="2200" value={fiscalYear} onChange={(event) => setFiscalYear(event.target.value)} /></label>
          <label>Document type<select value={documentType} onChange={(event) => setDocumentType(event.target.value)}><option value="">Unspecified</option><option value="annual_results">Annual results</option><option value="interim_results">Interim results</option><option value="annual_report">Annual report</option><option value="presentation">Presentation</option></select></label>
          <label>Language<input value={language} onChange={(event) => setLanguage(event.target.value)} placeholder="en, zh" /></label>
        </div>
        <footer>
          <button type="button" className="secondary-button" onClick={onCancel}>Cancel</button>
          <button type="submit" className="primary-button" disabled={busy}>{busy ? <LoaderCircle className="spin" size={16} /> : <Upload size={16} />}{busy ? "Queueing" : "Queue upload"}</button>
        </footer>
      </form>
    </div>
  );
}

function App() {
  const inputRef = useRef(null);
  const [documents, setDocuments] = useState([]);
  const [knowledgeBases, setKnowledgeBases] = useState([]);
  const [selectedKnowledgeBaseId, setSelectedKnowledgeBaseId] = useState(null);
  const [providers, setProviders] = useState([]);
  const [selectedProvider, setSelectedProvider] = useState("");
  const [selectedId, setSelectedId] = useState(null);
  const [tables, setTables] = useState([]);
  const [chunks, setChunks] = useState([]);
  const [selectedChunkId, setSelectedChunkId] = useState(null);
  const [selectedChunk, setSelectedChunk] = useState(null);
  const [chunkLoading, setChunkLoading] = useState(false);
  const [workspaceMode, setWorkspaceMode] = useState("knowledge_base");
  const [viewMode, setViewMode] = useState("tables");
  const [tableFilter, setTableFilter] = useState("pages");
  const [pageFilter, setPageFilter] = useState("all");
  const [busy, setBusy] = useState(false);
  const [pendingFile, setPendingFile] = useState(null);
  const [error, setError] = useState("");

  const selected = documents.find((document) => document.id === selectedId);
  const selectedKnowledgeBase = knowledgeBases.find((knowledgeBase) => knowledgeBase.id === selectedKnowledgeBaseId);
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

  async function loadDocuments(preferredId, knowledgeBaseId = selectedKnowledgeBaseId) {
    if (!knowledgeBaseId) return null;
    const result = await api(`/api/knowledge-bases/${knowledgeBaseId}/documents`);
    setDocuments(result);
    const currentId = preferredId || selectedId;
    const nextId = result.some((document) => document.id === currentId) ? currentId : result[0]?.id || null;
    setSelectedId(nextId);
    return nextId;
  }

  async function loadKnowledgeBases() {
    const result = await api("/api/knowledge-bases");
    setKnowledgeBases(result);
    setSelectedKnowledgeBaseId((current) => (
      result.some((knowledgeBase) => knowledgeBase.id === current) ? current : result[0]?.id || null
    ));
  }

  async function loadProviders() {
    const result = await api("/api/providers");
    setProviders(result);
    setSelectedProvider((current) => {
      if (result.some((provider) => provider.id === current && provider.available)) return current;
      return result.find((provider) => provider.default && provider.available)?.id
        || result.find((provider) => provider.available)?.id
        || "";
    });
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
    Promise.all([loadKnowledgeBases(), loadProviders()]).catch((err) => setError(err.message));
  }, []);

  useEffect(() => {
    loadDocuments(null, selectedKnowledgeBaseId).catch((err) => setError(err.message));
  }, [selectedKnowledgeBaseId]);

  useEffect(() => {
    if (selected?.status !== "READY") {
      setTables([]);
      setChunks([]);
      return;
    }
    Promise.all([loadTables(selectedId), loadChunks(selectedId)]).catch((err) => setError(err.message));
  }, [selectedId, selected?.status]);

  useEffect(() => {
    const hasActiveDocuments = documents.some((document) => !["READY", "FAILED"].includes(document.status));
    if (!selectedKnowledgeBaseId || !hasActiveDocuments) return undefined;
    const timer = window.setInterval(() => {
      Promise.all([
        loadDocuments(null, selectedKnowledgeBaseId),
        loadKnowledgeBases(),
      ]).catch((err) => setError(err.message));
    }, 3000);
    return () => window.clearInterval(timer);
  }, [documents, selectedKnowledgeBaseId]);

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

  async function uploadFile(file, metadata) {
    if (!file) return;
    setBusy(true);
    setError("");
    const body = new FormData();
    body.append("file", file);
    if (selectedProvider) body.append("provider", selectedProvider);
    Object.entries(metadata).forEach(([key, value]) => {
      if (value !== "") body.append(key, value);
    });
    try {
      const response = await api(`/api/knowledge-bases/${selectedKnowledgeBaseId}/documents`, { method: "POST", body });
      await loadDocuments(response.document.id, selectedKnowledgeBaseId);
      await loadKnowledgeBases();
      setWorkspaceMode("document");
      setPendingFile(null);
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
      if (inputRef.current) inputRef.current.value = "";
    }
  }

  async function createKnowledgeBase() {
    const name = window.prompt("Knowledge base name");
    if (!name?.trim()) return;
    try {
      const result = await api("/api/knowledge-bases", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: name.trim() }),
      });
      await loadKnowledgeBases();
      setSelectedKnowledgeBaseId(result.id);
    } catch (err) {
      setError(err.message);
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
          <label className="provider-picker">
            <Cpu size={15} />
            <span>Parser</span>
            <select value={selectedProvider} onChange={(event) => setSelectedProvider(event.target.value)}>
              {providers.length === 0 && <option value="">Loading models</option>}
              {providers.map((provider) => (
                <option key={provider.id} value={provider.id} disabled={!provider.available}>
                  {provider.name} · {provider.model}{provider.available ? "" : " · key required"}
                </option>
              ))}
            </select>
          </label>
          <button
            className="icon-button"
            title="Refresh documents"
            aria-label="Refresh documents"
            onClick={() => Promise.all([loadKnowledgeBases(), loadDocuments()]).catch((err) => setError(err.message))}
          >
            <RefreshCw size={18} />
          </button>
          <input
            ref={inputRef}
            type="file"
            accept="application/pdf,.pdf"
            hidden
            onChange={(event) => setPendingFile(event.target.files?.[0] || null)}
          />
          <button className="primary-button" onClick={() => inputRef.current?.click()} disabled={busy}>
            {busy ? <LoaderCircle className="spin" size={17} /> : <Upload size={17} />}
            {busy ? "Queueing" : "Upload PDF"}
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
          <div className="knowledge-base-picker">
            <div className="section-label"><Library size={15} /> Knowledge base</div>
            <div>
              <select value={selectedKnowledgeBaseId || ""} onChange={(event) => setSelectedKnowledgeBaseId(event.target.value)}>
                {knowledgeBases.map((knowledgeBase) => (
                  <option key={knowledgeBase.id} value={knowledgeBase.id}>
                    {knowledgeBase.name} ({knowledgeBase.ready_document_count}/{knowledgeBase.document_count})
                  </option>
                ))}
              </select>
              <button type="button" className="icon-button" onClick={createKnowledgeBase} title="Create knowledge base" aria-label="Create knowledge base"><Plus size={16} /></button>
            </div>
          </div>
          <button
            type="button"
            className={`knowledge-search-link ${workspaceMode === "knowledge_base" ? "selected" : ""}`}
            onClick={() => setWorkspaceMode("knowledge_base")}
          >
            <Search size={17} />
            <span>
              <strong>Search all documents</strong>
              <small>{selectedKnowledgeBase?.ready_document_count ?? 0} ready · {selectedKnowledgeBase?.document_count ?? 0} total</small>
            </span>
          </button>
          <div className="section-label"><Database size={15} /> Documents</div>
          <div className="document-list">
            {documents.map((document) => (
              <button
                key={document.id}
                className={`document-item ${selectedId === document.id ? "selected" : ""}`}
                onClick={() => {
                  setSelectedId(document.id);
                  setWorkspaceMode("document");
                }}
              >
                <FileText size={17} />
                <span>
                  <strong>{document.filename}</strong>
                  <small>{document.page_count ?? 0} pages · {document.metadata.tables ?? 0} tables</small>
                  <StatusBadge status={document.status} />
                  {document.metadata.table_repair_model && (
                    <small className="document-run">
                      {document.metadata.table_repair_model}
                      {document.metadata.parse_duration_seconds != null && ` · ${formatDuration(document.metadata.parse_duration_seconds)}`}
                    </small>
                  )}
                </span>
              </button>
            ))}
            {documents.length === 0 && <p className="empty-sidebar">No PDFs uploaded</p>}
          </div>
        </aside>

        <main className="main-content">
          {workspaceMode === "knowledge_base" && selectedKnowledgeBase ? (
            <>
              <section className="knowledge-search-header">
                <div>
                  <div className="eyebrow">Knowledge base retrieval</div>
                  <h2>{selectedKnowledgeBase.name}</h2>
                </div>
                <div className="knowledge-search-counts">
                  <span><strong>{selectedKnowledgeBase.ready_document_count}</strong> ready documents</span>
                  <span><strong>{selectedKnowledgeBase.document_count}</strong> total documents</span>
                </div>
              </section>
              <RetrievalDebugger
                key={`knowledge-base-${selectedKnowledgeBaseId}`}
                knowledgeBaseId={selectedKnowledgeBaseId}
                knowledgeBaseOnly
                onError={setError}
              />
            </>
          ) : !selected ? (
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
                  <div className="document-status-line">
                    <StatusBadge status={selected.status} />
                    {selected.metadata.table_repair_model && (
                      <span><Cpu size={13} /> {selected.metadata.table_repair_provider} · {selected.metadata.table_repair_model}</span>
                    )}
                    {selected.metadata.parse_duration_seconds != null && (
                      <span><Clock3 size={13} /> {formatDuration(selected.metadata.parse_duration_seconds)}</span>
                    )}
                    {selected.metadata.content_coverage != null && (
                      <span className={(selected.metadata.low_content_coverage_pages?.length ?? 0) > 0 ? "coverage-alert" : ""}>
                        <ScanSearch size={13} />
                        {Math.round(selected.metadata.content_coverage * 100)}% content coverage
                        {(selected.metadata.low_content_coverage_pages?.length ?? 0) > 0
                          && ` · ${selected.metadata.low_content_coverage_pages.length} flagged pages`}
                      </span>
                    )}
                  </div>
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
                <button type="button" role="tab" aria-selected={viewMode === "retrieval"} className={viewMode === "retrieval" ? "active" : ""} onClick={() => setViewMode("retrieval")}>
                  <GitMerge size={16} /> Retrieval
                </button>
              </div>

              {viewMode === "tables" && <><div className="table-toolbar">
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
              </>}
              {viewMode === "chunks" && (
                <ChunkExplorer
                  key={selectedId}
                  chunks={chunks}
                  selectedChunkId={selectedChunkId}
                  onSelect={setSelectedChunkId}
                  detail={selectedChunk}
                  loading={chunkLoading}
                />
              )}
              {viewMode === "retrieval" && (
                <RetrievalDebugger key={`${selectedId}-${selectedKnowledgeBaseId}`} documentId={selectedId} knowledgeBaseId={selectedKnowledgeBaseId} onError={setError} />
              )}
            </>
          )}
        </main>
      </div>
      {pendingFile && (
        <UploadDialog
          file={pendingFile}
          busy={busy}
          onCancel={() => {
            setPendingFile(null);
            if (inputRef.current) inputRef.current.value = "";
          }}
          onSubmit={uploadFile}
        />
      )}
    </div>
  );
}

createRoot(document.getElementById("root")).render(
  <React.StrictMode><App /></React.StrictMode>,
);
