export const fmtScore = (v: number | null | undefined) => (v == null ? "—" : v.toFixed(2));

export const fmtDelta = (v: number | null | undefined) =>
  v == null ? "" : `${v >= 0 ? "+" : "−"}${Math.abs(v).toFixed(2)}`;

export function fmtRuntime(s: number | null | undefined): string {
  if (s == null) return "—";
  if (s < 120) return `${Math.round(s)} s`;
  const m = Math.floor(s / 60);
  return m < 120 ? `${m} min ${Math.round(s % 60)} s` : `${(s / 3600).toFixed(1)} h`;
}

export const fmtCost = (v: number | null | undefined) => (v == null ? "—" : `$${v.toFixed(2)}`);

export function fmtDate(iso: string | null | undefined): string {
  if (!iso) return "—";
  const d = new Date(iso);
  return Number.isNaN(d.getTime())
    ? iso
    : d.toLocaleString(undefined, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
}

/** Approach → series color slot. Color follows the approach, never its rank. */
export const approachColor = (approach: string) =>
  ({
    baseline: "var(--series-baseline)",
    rag: "var(--series-rag)",
    long_context: "var(--series-long)",
    agent: "var(--series-agent)",
  })[approach] ?? "var(--text-muted)";

export const statusLabel: Record<string, string> = {
  queued: "Queued",
  running: "Running",
  completed: "Completed",
  failed: "Failed",
  waiting: "Waiting",
  skipped: "Skipped",
};
