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
  FlaskConical,
  GitMerge,
  Library,
  LoaderCircle,
  MessageSquareText,
  Plus,
  Play,
  RefreshCw,
  Search,
  ScanSearch,
  ShieldCheck,
  Sparkles,
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
            {(result.query_years?.length ?? 0) > 0 && <span><strong>Query years</strong> {result.query_years.join(", ")}</span>}
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

function QAWorkspace({ knowledgeBaseId, onError }) {
  const [question, setQuestion] = useState("2024到2025 AIA美国公司债变化如何，占公司债组合多少？");
  const [mode, setMode] = useState("PRODUCTION");
  const [company, setCompany] = useState("");
  const [fiscalYear, setFiscalYear] = useState("");
  const [snapshot, setSnapshot] = useState(null);
  const [stage, setStage] = useState("reranked_results");
  const [conversation, setConversation] = useState([]);
  const [retrieving, setRetrieving] = useState(false);
  const [answering, setAnswering] = useState(false);
  const [evidenceUpdated, setEvidenceUpdated] = useState(false);

  async function retrieveEvidence() {
    if (!question.trim()) return null;
    setRetrieving(true);
    onError("");
    try {
      const result = await api("/api/qa/retrieve", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          knowledge_base_id: knowledgeBaseId,
          query: question.trim(),
          mode,
          ...(company.trim() ? { company: company.trim() } : {}),
          ...(fiscalYear ? { fiscal_year: Number(fiscalYear) } : {}),
        }),
      });
      setSnapshot(result);
      setStage("reranked_results");
      setEvidenceUpdated(true);
      return result;
    } catch (err) {
      onError(err.message);
      return null;
    } finally {
      setRetrieving(false);
    }
  }

  async function generateAnswer(retrievalSnapshot, replaceLatest = false) {
    if (!retrievalSnapshot) return;
    setAnswering(true);
    onError("");
    try {
      const result = await api(`/api/qa/${retrievalSnapshot.retrieval_id}/answer`, { method: "POST" });
      const turn = { question: retrievalSnapshot.retrieval.original_query, ...result };
      setConversation((current) => {
        const latestUsesSameSnapshot = current.at(-1)?.retrieval_id === result.retrieval_id;
        return replaceLatest && latestUsesSameSnapshot
          ? [...current.slice(0, -1), turn]
          : [...current, turn];
      });
      setEvidenceUpdated(false);
    } catch (err) {
      onError(err.message);
    } finally {
      setAnswering(false);
    }
  }

  async function ask(event) {
    event.preventDefault();
    const retrievalSnapshot = await retrieveEvidence();
    if (retrievalSnapshot) await generateAnswer(retrievalSnapshot);
  }

  async function retrieveAgain() {
    await retrieveEvidence();
  }

  async function regenerateAnswer() {
    await generateAnswer(snapshot, conversation.length > 0);
  }

  const retrieval = snapshot?.retrieval;
  const hits = retrieval?.[stage] || [];
  return (
    <section className="qa-workspace">
      <div className="qa-left">
        <form className="qa-composer" onSubmit={ask}>
          <label htmlFor="qa-question">Question</label>
          <textarea id="qa-question" value={question} onChange={(event) => setQuestion(event.target.value)} rows={3} />
          <div className="qa-composer-controls">
            <div className="mode-control" aria-label="QA retrieval mode">
              {["BASELINE", "SEMANTIC", "PRODUCTION"].map((value) => (
                <button type="button" key={value} className={mode === value ? "active" : ""} onClick={() => setMode(value)}>{value[0] + value.slice(1).toLowerCase()}</button>
              ))}
            </div>
            <button type="submit" className="primary-button" disabled={retrieving || answering || !question.trim()}>
              {retrieving || answering ? <LoaderCircle className="spin" size={16} /> : <MessageSquareText size={16} />}
              {retrieving ? "Retrieving" : answering ? "Answering" : "Ask"}
            </button>
          </div>
          <div className="qa-filters">
            <label>Company<input value={company} onChange={(event) => setCompany(event.target.value)} placeholder="All companies" /></label>
            <label>Fiscal year<input type="number" min="1900" max="2200" value={fiscalYear} onChange={(event) => setFiscalYear(event.target.value)} placeholder="All years" /></label>
          </div>
        </form>

        <div className="conversation-heading"><MessageSquareText size={16} /><strong>Conversation</strong><span>{conversation.length}</span></div>
        <div className="conversation-list">
          {conversation.map((turn, index) => (
            <article className="conversation-turn" key={`${turn.retrieval_id}-${index}`}>
              <div className="question-bubble"><small>Question</small><p>{turn.question}</p></div>
              <div className={`answer-block ${turn.insufficient_evidence ? "insufficient" : ""}`}>
                <small>DeepSeek answer</small>
                <p>{turn.answer}</p>
                <div className="answer-meta"><span>{turn.model}</span>{turn.insufficient_evidence && <span>Insufficient evidence</span>}</div>
              </div>
              <div className="citation-list">
                <strong>Citations</strong>
                {turn.citations.map((citation) => (
                  <div key={`${citation.chunk_id}-${citation.evidence_number}`}>
                    <span>[E{citation.evidence_number}]</span>
                    <p>{citation.filename} · Page {citation.page}</p>
                    <code>{citation.chunk_id}</code>
                  </div>
                ))}
                {turn.citations.length === 0 && <p className="quiet">No citations returned.</p>}
              </div>
            </article>
          ))}
          {conversation.length === 0 && <div className="qa-empty"><MessageSquareText size={26} /><span>No answers yet</span></div>}
        </div>
        <div className="qa-answer-actions">
          <button type="button" className="secondary-button" onClick={retrieveAgain} disabled={retrieving || answering || !question.trim()}>
            {retrieving ? <LoaderCircle className="spin" size={15} /> : <Search size={15} />} Retrieve Again
          </button>
          <button type="button" className="secondary-button" onClick={regenerateAnswer} disabled={!snapshot || retrieving || answering}>
            {answering ? <LoaderCircle className="spin" size={15} /> : <RefreshCw size={15} />} Regenerate Answer
          </button>
          {evidenceUpdated && conversation.length > 0 && <span>Evidence updated; answer not regenerated.</span>}
        </div>
      </div>

      <aside className="qa-right">
        <div className="qa-debug-heading">
          <div><div className="eyebrow">Retrieval debug</div><strong>{retrieval ? `${retrieval.document_ids.length} documents searched` : "No retrieval run"}</strong></div>
          {retrieval?.query_years?.length > 0 && <span>Years {retrieval.query_years.join(", ")}</span>}
        </div>
        {retrieval && (
          <>
            <div className="stage-tabs" role="tablist" aria-label="QA retrieval stages">
              {RETRIEVAL_STAGES.map(([id, label]) => (
                <button type="button" role="tab" aria-selected={stage === id} className={stage === id ? "active" : ""} onClick={() => setStage(id)} key={id}>
                  {label}<span>{retrieval[id].length}</span>
                </button>
              ))}
            </div>
            <div className="qa-retrieval-results">
              {hits.map((hit) => <RetrievalHit hit={hit} key={hit.chunk_id} />)}
            </div>
          </>
        )}
        <section className="final-evidence">
          <header><strong>Final evidence passed to DeepSeek</strong><span>{snapshot?.final_evidence.length ?? 0} chunks</span></header>
          {(snapshot?.final_evidence || []).map((item) => (
            <article key={item.chunk_id}>
              <div><strong>[E{item.evidence_number}] {item.chunk_type}</strong><span>{item.filename} · Page {item.page}</span></div>
              <code>{item.chunk_id}</code>
              <pre>{item.content}</pre>
            </article>
          ))}
          {!snapshot && <div className="qa-empty"><Search size={24} /><span>Ask a question to retrieve evidence.</span></div>}
        </section>
      </aside>
    </section>
  );
}

function EvaluationResult({ result }) {
  const [stage, setStage] = useState("reranked_results");
  const hits = result.retrieval?.[stage] || [];
  const metrics = result.deterministic_metrics || {};
  return (
    <details className={`evaluation-result ${result.failure_stage ? "failed" : "passed"}`}>
      <summary>
        <span>{result.failure_stage ? <AlertTriangle size={15} /> : <CheckCircle2 size={15} />}</span>
        <strong>{result.question}</strong>
        <small>{result.failure_stage || "PASS"} · {formatDuration(result.latency_seconds)}</small>
      </summary>
      <div className="evaluation-result-body">
        <div className="evaluation-answer-grid">
          <section><small>Expected answer</small><p>{result.expected_answer}</p></section>
          <section><small>Actual answer</small><p>{result.answer || result.error || "No answer"}</p></section>
        </div>
        <div className="evaluation-score-strip">
          <span>Hit@1 <strong>{metrics.hit_at_1 ? "Yes" : "No"}</strong></span>
          <span>Numbers <strong>{metrics.number_accuracy ? "Pass" : "Fail"}</strong></span>
          <span>Units <strong>{metrics.unit_accuracy ? "Pass" : "Fail"}</strong></span>
          <span>Periods <strong>{metrics.period_accuracy ? "Pass" : "Fail"}</strong></span>
          <span>Citation recall <strong>{metrics.citation_recall == null ? "-" : `${Math.round(metrics.citation_recall * 100)}%`}</strong></span>
          <span>Judge <strong>{result.judge?.correctness == null ? "-" : `${Math.round(result.judge.correctness * 100)}%`}</strong></span>
        </div>
        {result.citations?.length > 0 && (
          <div className="evaluation-citations">
            {result.citations.map((citation) => <code key={citation.chunk_id}>{citation.filename} · p.{citation.page} · {citation.chunk_id}</code>)}
          </div>
        )}
        {result.retrieval && (
          <>
            <div className="stage-tabs" role="tablist">
              {RETRIEVAL_STAGES.map(([id, label]) => (
                <button type="button" key={id} className={stage === id ? "active" : ""} onClick={() => setStage(id)}>
                  {label}<span>{result.retrieval[id]?.length || 0}</span>
                </button>
              ))}
            </div>
            <div className="evaluation-trace">{hits.map((hit) => <RetrievalHit hit={hit} key={hit.chunk_id} />)}</div>
          </>
        )}
        <section className="evaluation-evidence">
          <h4>Final evidence passed to DeepSeek</h4>
          {(result.final_evidence || []).map((item) => (
            <div key={item.chunk_id}><code>[E{item.evidence_number}] {item.filename} · p.{item.page} · {item.chunk_id}</code><pre>{item.content}</pre></div>
          ))}
        </section>
        {result.judge?.rationale && <p className="judge-rationale"><strong>Qwen Judge:</strong> {result.judge.rationale}</p>}
      </div>
    </details>
  );
}

function EvaluationWorkspace({ knowledgeBaseId, knowledgeBaseName, onError }) {
  const importRef = useRef(null);
  const [tab, setTab] = useState("datasets");
  const [datasets, setDatasets] = useState([]);
  const [selectedDatasetId, setSelectedDatasetId] = useState(null);
  const [dataset, setDataset] = useState(null);
  const [runs, setRuns] = useState([]);
  const [selectedRunId, setSelectedRunId] = useState(null);
  const [run, setRun] = useState(null);
  const [comparisonRunId, setComparisonRunId] = useState("");
  const [comparisonRun, setComparisonRun] = useState(null);
  const [busy, setBusy] = useState(false);
  const [showCreate, setShowCreate] = useState(false);
  const [newDataset, setNewDataset] = useState({ name: "", description: "" });
  const [manual, setManual] = useState({ question: "", expected_answer: "", language: "en", expected_insufficient: false, filename: "", page: "", comparison_key: "", tags: "" });
  const [generator, setGenerator] = useState({ count: 10, language: "en", difficulty: "mixed", include_insufficient: true });
  const [runMode, setRunMode] = useState("PRODUCTION");

  async function loadDatasets(preferredId) {
    const rows = await api(`/api/evaluation/knowledge-bases/${knowledgeBaseId}/datasets`);
    setDatasets(rows);
    setSelectedDatasetId((current) => preferredId || (rows.some((item) => item.id === current) ? current : rows[0]?.id || null));
  }

  async function loadRuns(preferredId) {
    const rows = await api(`/api/evaluation/knowledge-bases/${knowledgeBaseId}/runs`);
    setRuns(rows);
    setSelectedRunId((current) => preferredId || (rows.some((item) => item.id === current) ? current : rows[0]?.id || null));
    return rows;
  }

  useEffect(() => {
    Promise.all([loadDatasets(), loadRuns()]).catch((error) => onError(error.message));
  }, [knowledgeBaseId]);

  useEffect(() => {
    if (!selectedDatasetId) { setDataset(null); return; }
    api(`/api/evaluation/datasets/${selectedDatasetId}`).then(setDataset).catch((error) => onError(error.message));
  }, [selectedDatasetId]);

  useEffect(() => {
    if (!selectedRunId) { setRun(null); return; }
    api(`/api/evaluation/runs/${selectedRunId}`).then(setRun).catch((error) => onError(error.message));
  }, [selectedRunId]);

  useEffect(() => {
    if (!comparisonRunId) { setComparisonRun(null); return; }
    api(`/api/evaluation/runs/${comparisonRunId}`).then(setComparisonRun).catch((error) => onError(error.message));
  }, [comparisonRunId]);

  useEffect(() => {
    if (!run || !["PENDING", "RUNNING"].includes(run.status)) return undefined;
    const timer = window.setInterval(async () => {
      try {
        const detail = await api(`/api/evaluation/runs/${run.id}`);
        setRun(detail);
        await loadRuns(detail.id);
      } catch (error) { onError(error.message); }
    }, 2500);
    return () => window.clearInterval(timer);
  }, [run?.id, run?.status]);

  async function act(callback) {
    setBusy(true);
    onError("");
    try { await callback(); } catch (error) { onError(error.message); } finally { setBusy(false); }
  }

  function createDataset(event) {
    event.preventDefault();
    act(async () => {
      const created = await api("/api/evaluation/datasets", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ knowledge_base_id: knowledgeBaseId, ...newDataset }),
      });
      setNewDataset({ name: "", description: "" });
      setShowCreate(false);
      await loadDatasets(created.id);
    });
  }

  function addManualCase(event) {
    event.preventDefault();
    act(async () => {
      const target = {};
      if (manual.filename.trim()) target.filename = manual.filename.trim();
      if (manual.page) target.page = Number(manual.page);
      if (manual.comparison_key.trim()) target.comparison_key = manual.comparison_key.trim();
      await api(`/api/evaluation/datasets/${dataset.id}/cases`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question: manual.question, expected_answer: manual.expected_answer, language: manual.language,
          expected_insufficient: manual.expected_insufficient,
          required_evidence: Object.keys(target).length ? [target] : [],
          tags: manual.tags.split(",").map((tag) => tag.trim()).filter(Boolean), status: "DRAFT",
        }),
      });
      setManual({ question: "", expected_answer: "", language: manual.language, expected_insufficient: false, filename: "", page: "", comparison_key: "", tags: "" });
      setDataset(await api(`/api/evaluation/datasets/${dataset.id}`));
      await loadDatasets(dataset.id);
    });
  }

  function generateCases() {
    act(async () => {
      await api(`/api/evaluation/datasets/${dataset.id}/generate`, {
        method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(generator),
      });
      setDataset(await api(`/api/evaluation/datasets/${dataset.id}`));
      await loadDatasets(dataset.id);
    });
  }

  function updateCase(caseId, changes) {
    act(async () => {
      await api(`/api/evaluation/cases/${caseId}`, {
        method: "PATCH", headers: { "Content-Type": "application/json" }, body: JSON.stringify(changes),
      });
      setDataset(await api(`/api/evaluation/datasets/${dataset.id}`));
      await loadDatasets(dataset.id);
    });
  }

  function importCases(file) {
    if (!file) return;
    act(async () => {
      const body = new FormData(); body.append("file", file);
      const result = await api(`/api/evaluation/datasets/${dataset.id}/import`, { method: "POST", body });
      if (result.errors.length) onError(`${result.imported} imported; ${result.rejected} rejected: ${result.errors[0]}`);
      setDataset(await api(`/api/evaluation/datasets/${dataset.id}`));
      await loadDatasets(dataset.id);
      importRef.current.value = "";
    });
  }

  function publishDataset() {
    act(async () => {
      await api(`/api/evaluation/datasets/${dataset.id}/publish`, { method: "POST" });
      setDataset(await api(`/api/evaluation/datasets/${dataset.id}`));
      await loadDatasets(dataset.id);
    });
  }

  function cloneDataset() {
    act(async () => {
      const cloned = await api(`/api/evaluation/datasets/${dataset.id}/clone`, { method: "POST" });
      await loadDatasets(cloned.id);
    });
  }

  function startRun() {
    act(async () => {
      const created = await api("/api/evaluation/runs", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ dataset_id: dataset.id, retrieval_mode: runMode }),
      });
      await loadRuns(created.id);
      setTab("runs");
    });
  }

  const metricKeys = [
    ["hit_at_1", "Hit@1"], ["hit_at_3", "Hit@3"], ["hit_at_5", "Hit@5"], ["mrr", "MRR"],
    ["number_accuracy", "Numbers"], ["citation_recall", "Citation recall"], ["judge_correctness", "Judge correctness"],
  ];
  return (
    <section className="evaluation-workspace">
      <header className="evaluation-header">
        <div><div className="eyebrow">Knowledge base evaluation</div><h2>{knowledgeBaseName}</h2></div>
        <div className="mode-control">
          <button type="button" className={tab === "datasets" ? "active" : ""} onClick={() => setTab("datasets")}>Test sets</button>
          <button type="button" className={tab === "runs" ? "active" : ""} onClick={() => setTab("runs")}>Runs</button>
        </div>
      </header>

      {tab === "datasets" && (
        <div className="evaluation-layout">
          <aside className="evaluation-list">
            <div className="evaluation-list-heading"><strong>Test sets</strong><button type="button" className="icon-button" onClick={() => setShowCreate((value) => !value)} title="Create test set"><Plus size={15} /></button></div>
            {showCreate && <form className="evaluation-create" onSubmit={createDataset}><input required placeholder="Test set name" value={newDataset.name} onChange={(event) => setNewDataset({ ...newDataset, name: event.target.value })} /><textarea placeholder="Description" value={newDataset.description} onChange={(event) => setNewDataset({ ...newDataset, description: event.target.value })} /><button className="primary-button" disabled={busy}>Create</button></form>}
            {datasets.map((item) => <button type="button" className={`evaluation-list-item ${item.id === selectedDatasetId ? "selected" : ""}`} onClick={() => setSelectedDatasetId(item.id)} key={item.id}><strong>{item.name} · v{item.version}</strong><span>{item.status} · {item.approved_case_count}/{item.case_count} approved</span></button>)}
            {datasets.length === 0 && <p className="quiet">No test sets yet.</p>}
          </aside>
          <div className="evaluation-detail">
            {!dataset ? <div className="qa-empty"><FlaskConical size={28} /><span>Create a test set to begin.</span></div> : (
              <>
                <div className="evaluation-dataset-title"><div><h3>{dataset.name} <span>v{dataset.version}</span></h3><p>{dataset.description || "No description"}</p></div><span className={`status ${dataset.status === "PUBLISHED" ? "ready" : "pending"}`}>{dataset.status}</span></div>
                <div className="evaluation-actions">
                  {dataset.status === "DRAFT" ? <><button className="secondary-button" onClick={generateCases} disabled={busy}><Sparkles size={15} /> Generate with Qwen</button><input ref={importRef} type="file" accept=".json,.csv" hidden onChange={(event) => importCases(event.target.files?.[0])} /><button className="secondary-button" onClick={() => importRef.current?.click()} disabled={busy}><Upload size={15} /> Import JSON/CSV</button><button className="primary-button" onClick={publishDataset} disabled={busy || dataset.approved_case_count === 0}><ShieldCheck size={15} /> Publish</button></> : <><button className="secondary-button" onClick={cloneDataset} disabled={busy}><Copy size={15} /> Clone new version</button><select value={runMode} onChange={(event) => setRunMode(event.target.value)}><option>PRODUCTION</option><option>SEMANTIC</option><option>BASELINE</option></select><button className="primary-button" onClick={startRun} disabled={busy}><Play size={15} /> Run evaluation</button></>}
                </div>
                {dataset.status === "DRAFT" && (
                  <div className="evaluation-builders">
                    <section><h4>AI generation</h4><div className="evaluation-generator"><label>Cases<input type="number" min="1" max="30" value={generator.count} onChange={(event) => setGenerator({ ...generator, count: Number(event.target.value) })} /></label><label>Language<select value={generator.language} onChange={(event) => setGenerator({ ...generator, language: event.target.value })}><option value="en">English</option><option value="zh">中文</option></select></label><label>Difficulty<select value={generator.difficulty} onChange={(event) => setGenerator({ ...generator, difficulty: event.target.value })}><option value="mixed">Mixed</option><option value="easy">Easy</option><option value="medium">Medium</option><option value="hard">Hard</option></select></label><label className="check-label"><input type="checkbox" checked={generator.include_insufficient} onChange={(event) => setGenerator({ ...generator, include_insufficient: event.target.checked })} /> Include insufficient</label></div></section>
                    <form onSubmit={addManualCase}><h4>Manual case</h4><textarea required placeholder="Question" value={manual.question} onChange={(event) => setManual({ ...manual, question: event.target.value })} /><textarea required placeholder="Expected answer" value={manual.expected_answer} onChange={(event) => setManual({ ...manual, expected_answer: event.target.value })} /><div><select value={manual.language} onChange={(event) => setManual({ ...manual, language: event.target.value })}><option value="en">English</option><option value="zh">中文</option></select><input placeholder="Filename" value={manual.filename} onChange={(event) => setManual({ ...manual, filename: event.target.value })} /><input type="number" min="1" placeholder="Page" value={manual.page} onChange={(event) => setManual({ ...manual, page: event.target.value })} /><input placeholder="Comparison key" value={manual.comparison_key} onChange={(event) => setManual({ ...manual, comparison_key: event.target.value })} /><input placeholder="Tags, comma separated" value={manual.tags} onChange={(event) => setManual({ ...manual, tags: event.target.value })} /></div><label className="manual-insufficient"><input type="checkbox" checked={manual.expected_insufficient} onChange={(event) => setManual({ ...manual, expected_insufficient: event.target.checked })} /> Expected answer is insufficient evidence</label><button className="secondary-button" disabled={busy}><Plus size={15} /> Add draft case</button></form>
                  </div>
                )}
                <div className="evaluation-case-list">
                  {dataset.cases.map((item, index) => <article key={item.id}><div className="case-index">{String(index + 1).padStart(2, "0")}</div><div><strong>{item.question}</strong><p>{item.expected_answer}</p><small>{item.source} · {item.language} · {item.difficulty} · {(item.tags || []).join(", ") || "untagged"}</small>{item.required_evidence.map((target, targetIndex) => <code key={targetIndex}>{target.filename || "any file"} · p.{target.page || "any"} · {target.comparison_key || target.chunk_id || "locator pending"}</code>)}</div>{dataset.status === "DRAFT" && <button className={`case-review ${item.status === "APPROVED" ? "approved" : ""}`} onClick={() => updateCase(item.id, { status: item.status === "APPROVED" ? "DRAFT" : "APPROVED" })}>{item.status === "APPROVED" ? "Approved" : "Approve"}</button>}</article>)}
                </div>
              </>
            )}
          </div>
        </div>
      )}

      {tab === "runs" && (
        <div className="evaluation-layout">
          <aside className="evaluation-list">
            <div className="evaluation-list-heading"><strong>Evaluation runs</strong><RefreshCw size={15} /></div>
            {runs.map((item) => <button type="button" className={`evaluation-list-item ${item.id === selectedRunId ? "selected" : ""}`} onClick={() => setSelectedRunId(item.id)} key={item.id}><strong>{item.retrieval_mode} · {item.status}</strong><span>{item.progress}% · {new Date(item.created_at).toLocaleString()}</span></button>)}
            {runs.length === 0 && <p className="quiet">No runs yet. Publish a test set first.</p>}
          </aside>
          <div className="evaluation-detail">
            {!run ? <div className="qa-empty"><Play size={28} /><span>Select a run.</span></div> : <>
              <div className="evaluation-run-heading"><div><div className="eyebrow">{run.retrieval_mode} · {run.status}</div><h3>Run {run.id.slice(0, 8)}</h3></div><div className="run-progress"><span style={{ width: `${run.progress}%` }} /></div></div>
              {run.error && <div className="warning"><AlertTriangle size={16} /><p>{run.error}</p></div>}
              {Object.keys(run.metrics || {}).length > 0 && <div className="evaluation-metrics">{metricKeys.map(([key, label]) => <div key={key}><span>{label}</span><strong>{run.metrics[key] == null ? "-" : `${Math.round(run.metrics[key] * 100)}%`}</strong></div>)}</div>}
              {run.status === "COMPLETE" && runs.filter((item) => item.status === "COMPLETE" && item.id !== run.id).length > 0 && <div className="run-compare"><label>Compare with<select value={comparisonRunId} onChange={(event) => setComparisonRunId(event.target.value)}><option value="">Select another run</option>{runs.filter((item) => item.status === "COMPLETE" && item.id !== run.id).map((item) => <option key={item.id} value={item.id}>{item.retrieval_mode} · {item.id.slice(0, 8)}</option>)}</select></label>{comparisonRun && <div>{metricKeys.map(([key, label]) => <span key={key}><small>{label}</small><strong>{Math.round((run.metrics[key] || 0) * 100)}%</strong><em>{Math.round((comparisonRun.metrics[key] || 0) * 100)}%</em></span>)}</div>}</div>}
              <div className="evaluation-results">{(run.results || []).map((result) => <EvaluationResult result={result} key={result.id} />)}</div>
            </>}
          </div>
        </div>
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
          <button
            type="button"
            className={`knowledge-search-link ${workspaceMode === "evaluation" ? "selected" : ""}`}
            onClick={() => setWorkspaceMode("evaluation")}
          >
            <FlaskConical size={17} />
            <span><strong>Evaluation</strong><small>Test sets, runs and failure analysis</small></span>
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
          {workspaceMode === "evaluation" && selectedKnowledgeBase ? (
            <EvaluationWorkspace
              key={`evaluation-${selectedKnowledgeBaseId}`}
              knowledgeBaseId={selectedKnowledgeBaseId}
              knowledgeBaseName={selectedKnowledgeBase.name}
              onError={setError}
            />
          ) : workspaceMode === "knowledge_base" && selectedKnowledgeBase ? (
            <>
              <section className="knowledge-search-header">
                <div>
                  <div className="eyebrow">Knowledge base QA</div>
                  <h2>{selectedKnowledgeBase.name}</h2>
                </div>
                <div className="knowledge-search-counts">
                  <span><strong>{selectedKnowledgeBase.ready_document_count}</strong> ready documents</span>
                  <span><strong>{selectedKnowledgeBase.document_count}</strong> total documents</span>
                </div>
              </section>
              <QAWorkspace
                key={`knowledge-base-${selectedKnowledgeBaseId}`}
                knowledgeBaseId={selectedKnowledgeBaseId}
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
