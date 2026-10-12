import { Link, useSearchParams } from "react-router-dom";
import { api } from "../api";
import { Empty, ErrorNote, Loading, Panel } from "../components/ui";
import { useAsync, useMeta } from "../hooks";
import { approachColor } from "../format";

export default function FailuresPage() {
  const meta = useMeta();
  const [params, setParams] = useSearchParams();
  const task = params.get("task") ?? "";
  const approach = params.get("approach") ?? "";
  const category = params.get("category") ?? "";
  const run = params.get("run") ?? "";

  const report = useAsync(
    () => api.failures({ task, approach, category, run_id: run }),
    [task, approach, category, run],
  );

  const update = (key: string, value: string) => {
    const next = new URLSearchParams(params);
    if (value) next.set(key, value);
    else next.delete(key);
    setParams(next);
  };

  const counts = report.data?.counts ?? {};
  const total = Object.values(counts).reduce((a, b) => a + b, 0);
  const max = Math.max(1, ...Object.values(counts));
  const approachLabel = (id: string) => meta.approaches.find((a) => a.id === id)?.label ?? id;
  const taskLabel = (id: string) => meta.tasks.find((t) => t.id === id)?.label ?? id;

  return (
    <div className="grid-configure">
      <Panel
        title="Failure taxonomy"
        subtitle={`${total} analysed failure${total === 1 ? "" : "s"}${approach ? ` for ${approachLabel(approach)}` : ""}`}
        className="sidebar"
      >
        <ul className="taxonomy">
          {meta.failure_categories.map((c) => {
            const n = counts[c.id] ?? 0;
            const active = category === c.id;
            return (
              <li key={c.id}>
                <button
                  type="button"
                  className={`taxonomy-item${active ? " active" : ""}`}
                  onClick={() => update("category", active ? "" : c.id)}
                  aria-pressed={active}
                  title={c.description}
                >
                  <span className="taxonomy-head">
                    <span>{c.label}</span>
                    <span className="taxonomy-count">{n}</span>
                  </span>
                  <span className="bar-track thin">
                    <span className="bar neutral" style={{ width: `${(n / max) * 100}%` }} />
                  </span>
                </button>
              </li>
            );
          })}
        </ul>
        <p className="muted small">
          Retrieval problems (missing or irrelevant context) are separated from reasoning problems where the right
          files were available.
        </p>
      </Panel>

      <Panel
        title="Representative failures"
        className="main-panel"
        actions={
          <div className="filters">
            <select aria-label="Task" value={task} onChange={(e) => update("task", e.target.value)}>
              <option value="">All tasks</option>
              {meta.tasks.map((t) => (
                <option key={t.id} value={t.id}>
                  {t.label}
                </option>
              ))}
            </select>
            <select aria-label="Approach" value={approach} onChange={(e) => update("approach", e.target.value)}>
              <option value="">All approaches</option>
              {meta.approaches.map((a) => (
                <option key={a.id} value={a.id}>
                  {a.label}
                </option>
              ))}
            </select>
            {run && (
              <button type="button" className="chip" onClick={() => update("run", "")}>
                run: {run} ✕
              </button>
            )}
          </div>
        }
      >
        {report.error && <ErrorNote error={report.error} />}
        {!report.data && !report.error && <Loading />}
        {report.data?.items.length === 0 && <Empty>No failures match these filters.</Empty>}
        <ul className="failure-list">
          {report.data?.items.map((f) => {
            const missed = f.expected.filter((p) => !f.retrieved.includes(p));
            return (
              <li key={`${f.run_id}:${f.instance_id}`} className="failure">
                <div className="failure-head">
                  <span className="swatch" style={{ background: approachColor(f.approach) }} aria-hidden />
                  <span className="strong">{approachLabel(f.approach)}</span>
                  <span className="muted">· {taskLabel(f.task)} · {f.instance_id}</span>
                  <span className="finding-tag">
                    {meta.failure_categories.find((c) => c.id === f.category)?.label ?? f.category}
                  </span>
                </div>
                <p>{f.summary}</p>
                {(f.retrieved.length > 0 || f.expected.length > 0) && (
                  <div className="paths">
                    {f.retrieved.map((p) => (
                      <code key={`r-${p}`} className={f.expected.includes(p) ? "path hit" : "path"}>
                        {p}
                      </code>
                    ))}
                    {missed.map((p) => (
                      <code key={`m-${p}`} className="path missed" title="Expected but not supplied to the model">
                        {p} (missed)
                      </code>
                    ))}
                  </div>
                )}
                <Link className="link-inline" to={`/runs/${encodeURIComponent(f.run_id)}`}>
                  Open run ›
                </Link>
                {f.url && (
                  <a className="link-inline" href={f.url} target="_blank" rel="noreferrer">
                    {" "}View CI job ›
                  </a>
                )}
              </li>
            );
          })}
        </ul>
      </Panel>
    </div>
  );
}
