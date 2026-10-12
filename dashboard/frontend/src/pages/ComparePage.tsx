import { useEffect, useMemo, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { api, type CompareReport, type CompareRow, type Option } from "../api";
import { Empty, ErrorNote, Loading, Panel } from "../components/ui";
import { useAsync, useMeta } from "../hooks";
import { approachColor, fmtCost, fmtDelta, fmtRuntime, fmtScore } from "../format";

export default function ComparePage() {
  const meta = useMeta();
  const [params, setParams] = useSearchParams();
  const runs = useAsync(() => api.runs({ status: "completed" }), []);

  // Default to the task with the most completed approaches.
  const tasksWithResults = useMemo(() => {
    const counts = new Map<string, Set<string>>();
    for (const r of runs.data ?? []) {
      if (!counts.has(r.task)) counts.set(r.task, new Set());
      counts.get(r.task)!.add(r.approach);
    }
    return counts;
  }, [runs.data]);
  const fallbackTask =
    [...tasksWithResults.entries()].sort((a, b) => b[1].size - a[1].size)[0]?.[0] ?? meta.tasks[0].id;
  const task = params.get("task") ?? fallbackTask;
  const model = params.get("model") ?? undefined;

  const report = useAsync(() => api.compare(task, model), [task, model]);
  const [selected, setSelected] = useState<string>();
  useEffect(() => setSelected(undefined), [task, model]);
  const rows = report.data?.rows ?? [];
  const selectedRow = rows.find((r) => r.approach === selected) ?? rows[rows.length - 1];
  const hasRuntime = rows.some((r) => r.runtime_s != null);
  const hasCost = rows.some((r) => r.cost_usd != null);

  return (
    <div className="grid-single">
      <Panel
        title="Same task, same evaluator, different context strategies"
        actions={
          <>
            <label className="inline-select">
              <span className="sr-only">Benchmark task</span>
              <select value={task} onChange={(e) => setParams({ task: e.target.value })}>
                {meta.tasks.map((t) => (
                  <option key={t.id} value={t.id}>
                    {t.label}
                    {tasksWithResults.has(t.id) ? ` (${tasksWithResults.get(t.id)!.size})` : " (no results)"}
                  </option>
                ))}
              </select>
            </label>
            {report.data && report.data.models.length > 0 && (
              <label className="inline-select">
                <span className="sr-only">Model</span>
                <select
                  value={report.data.model ?? ""}
                  onChange={(e) => setParams({ task, model: e.target.value })}
                >
                  {report.data.models.map((m) => (
                    <option key={m} value={m}>
                      {m}
                    </option>
                  ))}
                </select>
              </label>
            )}
            {selectedRow && (
              <Link className="inspect" to={`/runs/${encodeURIComponent(selectedRow.run_id)}`}>
                INSPECT TRACE ›
              </Link>
            )}
          </>
        }
      >
        {report.error && <ErrorNote error={report.error} />}
        {!report.data && !report.error && <Loading />}
        {report.data && rows.length === 0 && (
          <Empty>
            No completed runs for {report.data.task_label} yet. <Link to="/configure">Configure one</Link>.
          </Empty>
        )}
        {report.data && rows.length > 0 && (
          <>
            <div className="table-wrap">
              <table className="results-table">
                <thead>
                  <tr>
                    <th>Approach</th>
                    <th className="num">{report.data.metric}</th>
                    <th className="num">vs baseline</th>
                    {hasRuntime && <th className="num">Runtime</th>}
                    {hasCost && <th className="num">Cost</th>}
                    <th>Key observation</th>
                  </tr>
                </thead>
                <tbody>
                  {rows.map((r) => (
                    <tr
                      key={r.approach}
                      className={r === selectedRow ? "selected" : ""}
                      onClick={() => setSelected(r.approach)}
                      tabIndex={0}
                      onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && setSelected(r.approach)}
                      aria-selected={r === selectedRow}
                    >
                      <td>
                        <span className="swatch" style={{ background: approachColor(r.approach) }} aria-hidden />
                        {r.approach_label}
                      </td>
                      <td className="num strong">{fmtScore(r.score)}</td>
                      <td className={`num ${r.delta_vs_baseline == null ? "" : r.delta_vs_baseline >= 0 ? "up" : "down"}`}>
                        {fmtDelta(r.delta_vs_baseline) || "—"}
                      </td>
                      {hasRuntime && <td className="num">{fmtRuntime(r.runtime_s)}</td>}
                      {hasCost && <td className="num">{fmtCost(r.cost_usd)}</td>}
                      <td className="observation">{r.observation ?? "—"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            <div className="charts">
              <BarChart title={report.data.metric} rows={rows} value={(r) => r.score} format={fmtScore} domainMax={1} />
              {hasRuntime && <BarChart title="Runtime" rows={rows} value={(r) => r.runtime_s} format={fmtRuntime} />}
              {hasCost && <BarChart title="Cost per run" rows={rows} value={(r) => r.cost_usd} format={fmtCost} />}
            </div>

            {report.data.leaderboard.length > 1 && (
              <Leaderboard
                report={report.data}
                approaches={meta.approaches.filter((a) =>
                  report.data!.leaderboard.some((row) => row.cells[a.id]),
                )}
                onPick={(m) => setParams({ task, model: m })}
              />
            )}

            {report.data.example_finding && (
              <div className="finding">
                <div className="finding-title">
                  Example finding
                  <span className="finding-tag">
                    {meta.failure_categories.find((c) => c.id === report.data!.example_finding!.category)?.label}
                  </span>
                </div>
                <p>{report.data.example_finding.summary}</p>
                <Link className="link-inline" to={`/failures?task=${task}`}>
                  All failures for {report.data.task_label} ›
                </Link>
              </div>
            )}
          </>
        )}
      </Panel>
    </div>
  );
}

/** Every model on this task, one column per approach; best score per column in bold. */
function Leaderboard({ report, approaches, onPick }: {
  report: CompareReport;
  approaches: Option[];
  onPick: (model: string) => void;
}) {
  const best = Object.fromEntries(
    approaches.map((a) => [a.id, Math.max(...report.leaderboard.map((row) => row.cells[a.id]?.score ?? -Infinity))]),
  );
  return (
    <div className="leaderboard">
      <div className="section-label">ALL MODELS · {report.metric.toUpperCase()}</div>
      <div className="table-wrap">
        <table className="results-table">
          <thead>
            <tr>
              <th>Model</th>
              {approaches.map((a) => (
                <th key={a.id} className="num">
                  <span className="swatch" style={{ background: approachColor(a.id) }} aria-hidden />
                  {a.label}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {report.leaderboard.map((row) => (
              <tr
                key={row.model}
                className={row.model === report.model ? "selected" : ""}
                onClick={() => onPick(row.model)}
                tabIndex={0}
                onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && onPick(row.model)}
                aria-selected={row.model === report.model}
              >
                <td>{row.model}</td>
                {approaches.map((a) => {
                  const cell = row.cells[a.id];
                  return (
                    <td key={a.id} className={`num${cell && cell.score === best[a.id] ? " strong" : ""}`}>
                      {cell ? (
                        <Link to={`/runs/${encodeURIComponent(cell.run_id)}`} onClick={(e) => e.stopPropagation()}>
                          {fmtScore(cell.score)}
                        </Link>
                      ) : (
                        "—"
                      )}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

/** Horizontal bars, one per approach, direct-labelled; hover shows the exact value. */
function BarChart({ title, rows, value, format, domainMax }: {
  title: string;
  rows: CompareRow[];
  value: (r: CompareRow) => number | null;
  format: (v: number | null) => string;
  domainMax?: number;
}) {
  const max = Math.max(domainMax ?? 0, ...rows.map((r) => value(r) ?? 0)) || 1;
  return (
    <figure className="bar-chart">
      <figcaption>{title}</figcaption>
      {rows.map((r) => {
        const v = value(r);
        return (
          <div key={r.approach} className="bar-row" title={`${r.approach_label}: ${format(v)}`}>
            <span className="bar-label">{r.approach_label}</span>
            <span className="bar-track">
              {v != null && (
                <span
                  className="bar"
                  style={{ width: `${Math.max(1, (v / max) * 100)}%`, background: approachColor(r.approach) }}
                />
              )}
            </span>
            <span className="bar-value">{format(v)}</span>
          </div>
        );
      })}
    </figure>
  );
}
