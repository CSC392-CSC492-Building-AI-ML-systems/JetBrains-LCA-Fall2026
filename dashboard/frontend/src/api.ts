// Typed client for the FastAPI backend (dashboard/backend/app/main.py).

export type RunStatus = "queued" | "running" | "completed" | "failed";
export type StageStatus = "waiting" | "running" | "completed" | "failed" | "skipped";

export interface Option {
  id: string;
  label: string;
  description: string;
}

export interface TaskInfo {
  id: string;
  label: string;
  metric: string;
  higher_is_better: boolean;
}

export interface Meta {
  tasks: TaskInfo[];
  approaches: Option[];
  failure_categories: Option[];
  results_dir: string;
  sources: { id: string; label: string; path: string; runs: number }[];
  runner_configured: boolean;
}

export interface Stage {
  name: string;
  status: StageStatus;
  detail?: string | null;
}

export interface Evidence {
  path: string;
  kind: "retrieved" | "expected" | "both";
  score?: number | null;
  note?: string | null;
}

export interface RunSummary {
  id: string;
  name: string;
  task: string;
  task_label: string;
  approach: string;
  approach_label: string;
  model: string | null;
  status: RunStatus;
  created_at: string | null;
  score: number | null;
  runtime_s: number | null;
  cost_usd: number | null;
  failure_count: number;
  /** "dashboard" for launched runs, else the repo folder the run was read from */
  source: string;
}

export interface RunDetail extends RunSummary {
  repo_version: string | null;
  started_at: string | null;
  finished_at: string | null;
  settings: Record<string, unknown>;
  metrics: Record<string, number | null>;
  tokens: { input?: number | null; output?: number | null } | null;
  instances: number | null;
  observation: string | null;
  stages: Stage[];
  evidence: Evidence[];
  config_yaml: string | null;
  source_path: string | null;
}

export interface Failure {
  run_id: string;
  task: string;
  approach: string;
  instance_id: string;
  category: string;
  summary: string;
  retrieved: string[];
  expected: string[];
  url?: string | null;
}

export interface FailureReport {
  categories: Option[];
  counts: Record<string, number>;
  counts_by_approach: Record<string, Record<string, number>>;
  items: Failure[];
}

export interface CompareRow {
  approach: string;
  approach_label: string;
  run_id: string;
  model: string | null;
  score: number | null;
  delta_vs_baseline: number | null;
  runtime_s: number | null;
  cost_usd: number | null;
  observation: string | null;
  failure_count: number;
}

export interface CompareReport {
  task: string;
  task_label: string;
  metric: string;
  models: string[];
  model: string | null;
  rows: CompareRow[];
  leaderboard: { model: string; cells: Record<string, { run_id: string; score: number | null }> }[];
  example_finding: Failure | null;
}

export interface TreeNode {
  name: string;
  path: string;
  type: "dir" | "file";
  children?: TreeNode[] | null;
}

export interface ConfigFile {
  name: string;
  path: string;
  content: string;
  task: string | null;
}

export interface LaunchRequest {
  task: string;
  approach: string;
  model: string;
  repo_version: string;
  name?: string;
  settings: Record<string, unknown>;
}

export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message);
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`/api${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...init?.headers },
  });
  if (!res.ok) {
    let message = res.statusText;
    try {
      const body = await res.json();
      message = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch {
      /* keep statusText */
    }
    throw new ApiError(res.status, message);
  }
  return res.json() as Promise<T>;
}

function qs(params: Record<string, string | number | undefined | null>): string {
  const entries = Object.entries(params).filter(([, v]) => v !== undefined && v !== null && v !== "");
  return entries.length ? `?${new URLSearchParams(entries.map(([k, v]) => [k, String(v)]))}` : "";
}

export const api = {
  meta: () => request<Meta>("/meta"),
  projectTree: () => request<TreeNode[]>("/project-tree"),
  configs: () => request<ConfigFile[]>("/configs"),
  saveDraft: (body: LaunchRequest) =>
    request<ConfigFile>("/configs/drafts", { method: "POST", body: JSON.stringify(body) }),
  runs: (filters: { task?: string; approach?: string; status?: string } = {}) =>
    request<RunSummary[]>(`/runs${qs(filters)}`),
  run: (id: string) => request<RunDetail>(`/runs/${encodeURIComponent(id)}`),
  log: (id: string, offset = 0) =>
    request<{ lines: string[]; next_offset: number }>(`/runs/${encodeURIComponent(id)}/log${qs({ offset })}`),
  launch: (body: LaunchRequest) => request<RunDetail>("/runs", { method: "POST", body: JSON.stringify(body) }),
  compare: (task: string, model?: string) => request<CompareReport>(`/compare${qs({ task, model })}`),
  failures: (filters: { task?: string; approach?: string; category?: string; run_id?: string } = {}) =>
    request<FailureReport>(`/failures${qs(filters)}`),
};
