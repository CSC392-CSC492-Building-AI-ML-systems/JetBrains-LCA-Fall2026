import { useEffect, useRef, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { api, type Evidence, type RunDetail } from "../api";
import { Empty, ErrorNote, Loading, Panel, StatusDot, StatusPill } from "../components/ui";
import { useAsync, useMeta } from "../hooks";
import { approachColor, fmtCost, fmtDate, fmtRuntime, fmtScore, statusLabel } from "../format";

const POLL_MS = 2000;
const isLive = (status?: string) => status === "running" || status === "queued";

const EVIDENCE_LABEL: Record<Evidence["kind"], string> = {
  retrieved: "retrieved",
  expected: "expected · not retrieved",
  both: "retrieved · expected",
};

export default function RunsPage() {
  const { runId } = useParams();
  const navigate = useNavigate();
  const runs = useAsync(() => api.runs(), [], { poll: true, pollMs: 5000 });

  // With no run in the URL, open the most relevant one: something live, else the newest.
  useEffect(() => {
    if (!runId && runs.data?.length) {
      const pick = runs.data.find((r) => isLive(r.status)) ?? runs.data[0];
      navigate(`/runs/${encodeURIComponent(pick.id)}`, { replace: true });
    }
  }, [runId, runs.data, navigate]);

  return (
    <div className="grid-runs">
      <Panel title="Runs" className="sidebar run-list-panel">
        {runs.error && <ErrorNote error={runs.error} />}
        {!runs.data && !runs.error && <Loading />}
        {runs.data?.length === 0 && (
          <Empty>
            No runs yet. <Link to="/configure">Configure one</Link>.
          </Empty>
        )}
        <ul className="run-list">
          {runs.data?.map((r) => (
            <li key={r.id}>
              <Link to={`/runs/${encodeURIComponent(r.id)}`} className={`run-link${r.id === runId ? " selected" : ""}`}>
                <span className="swatch" style={{ background: approachColor(r.approach) }} aria-hidden />
                <span className="run-link-body">
                  <span className="run-link-title">
                    {r.approach_label} · {r.task_label}
                  </span>
                  {r.model && <span className="run-link-meta">{r.model}</span>}
                  <span className="run-link-meta">
                    <StatusDot status={r.status} /> {statusLabel[r.status]} · {fmtDate(r.created_at)}
                    {r.score != null && ` · ${fmtScore(r.score)}`}
                  </span>
                </span>
              </Link>
            </li>
          ))}
        </ul>
      </Panel>
      {runId ? <RunView key={runId} runId={runId} onChange={runs.reload} /> : <div />}
    </div>
  );
}

function RunView({ runId, onChange }: { runId: string; onChange: () => void }) {
  const meta = useMeta();
  // Poll while the run is queued/running; `live` lags the data by one render.
  const [live, setLive] = useState(true);
  const run = useAsync(() => api.run(runId), [runId], { poll: live, pollMs: POLL_MS });
  const log = useAsync(() => api.log(runId), [runId], { poll: live, pollMs: POLL_MS });
  const detail = run.data;
  useEffect(() => {
    if (detail) setLive(isLive(detail.status));
  }, [detail]);

  // Refresh the run list once when this run finishes.
  const prevStatus = useRef(detail?.status);
  useEffect(() => {
    if (isLive(prevStatus.current) && detail && !isLive(detail.status)) onChange();
    prevStatus.current = detail?.status;
  }, [detail, onChange]);

  if (run.error && !detail) return <ErrorNote error={run.error} />;
  if (!detail) return <Loading />;

  const metric = meta.tasks.find((t) => t.id === detail.task)?.metric ?? "Score";
  return (
    <>
      <Panel title={detail.name} actions={<StatusPill status={detail.status} />} className="main-panel">
        <ol className="stages">
          {detail.stages.map((s) => (
            <li key={s.name} className={`stage stage-${s.status}`}>
              <div>
                <div className="stage-name">{s.name}</div>
                <div className="stage-status">{s.detail ?? statusLabel[s.status]}</div>
              </div>
              <StatusDot status={s.status} />
            </li>
          ))}
          {detail.stages.length === 0 && <Empty>This run did not record stages.</Empty>}
        </ol>

        <div className="stat-row">
          <Stat label={metric} value={fmtScore(detail.score)} />
          {/* Repo results rarely record runtime, cost or tokens; skip tiles with nothing to show. */}
          {(detail.source === "dashboard" || detail.runtime_s != null) && (
            <Stat label="Runtime" value={fmtRuntime(detail.runtime_s)} />
          )}
          {(detail.source === "dashboard" || detail.cost_usd != null) && (
            <Stat label="Cost" value={fmtCost(detail.cost_usd)} />
          )}
          {(detail.source === "dashboard" || detail.tokens?.input != null) && (
            <Stat
              label="Tokens in / out"
              value={
                detail.tokens?.input != null
                  ? `${(detail.tokens.input / 1000).toFixed(0)}k / ${((detail.tokens.output ?? 0) / 1000).toFixed(1)}k`
                  : "—"
              }
            />
          )}
        </div>

        <RunFacts run={detail} />
      </Panel>

      <Panel title="Evidence used" subtitle="Inspect supplied context before judging a failure." className="evidence-panel">
        <ul className="evidence-list">
          {detail.evidence.map((e) => (
            <li key={e.path} className={`evidence evidence-${e.kind}`}>
              <div className="evidence-path">{e.path}</div>
              <div className="evidence-kind">
                {EVIDENCE_LABEL[e.kind]}
                {e.score != null && ` · sim ${e.score.toFixed(2)}`}
              </div>
            </li>
          ))}
          {detail.evidence.length === 0 && <Empty>No context trace recorded for this run.</Empty>}
        </ul>
        {detail.failure_count > 0 && (
          <Link className="link-inline" to={`/failures?run=${encodeURIComponent(detail.id)}`}>
            {detail.failure_count} analysed failure{detail.failure_count > 1 ? "s" : ""} ›
          </Link>
        )}
        <div className="console" aria-live="polite">
          <div className="section-label">RUN CONSOLE</div>
          {(log.data?.lines ?? []).slice(-200).map((line, i, arr) => (
            <div key={i} className={`console-line${i === arr.length - 1 && isLive(detail.status) ? " live" : ""}`}>
              {line}
            </div>
          ))}
          {log.data?.lines.length === 0 && <div className="muted">No output yet.</div>}
        </div>
      </Panel>
    </>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="stat">
      <div className="stat-label">{label}</div>
      <div className="stat-value">{value}</div>
    </div>
  );
}

function RunFacts({ run }: { run: RunDetail }) {
  const fmtValue = (v: unknown) =>
    typeof v === "number" ? (Number.isInteger(v) ? String(v) : v.toFixed(4)) : v == null ? "—" : String(v);
  const facts: [string, string][] = [
    ["Model", run.model ?? "—"],
    ["Repository", run.repo_version ?? "—"],
    ["Instances", run.instances != null ? String(run.instances) : "—"],
    [run.source === "dashboard" ? "Started" : "Files written", fmtDate(run.started_at ?? run.created_at)],
    ["Run id", run.id],
    ["Read from", run.source_path ?? "—"],
    ...Object.entries(run.metrics)
      .filter(([k]) => k !== "score")
      .map(([k, v]): [string, string] => [k.replace(/_/g, " "), fmtValue(v)]),
    ...Object.entries(run.settings).map(([k, v]): [string, string] => [
      k.replace(/_/g, " "),
      Array.isArray(v) ? v.join(", ") : fmtValue(v),
    ]),
  ];
  return (
    <div className="facts">
      <dl>
        {facts.map(([k, v]) => (
          <div key={k}>
            <dt>{k}</dt>
            <dd>{v}</dd>
          </div>
        ))}
      </dl>
      {run.config_yaml && (
        <details className="yaml-preview">
          <summary>config.yaml</summary>
          <pre>{run.config_yaml}</pre>
        </details>
      )}
    </div>
  );
}
