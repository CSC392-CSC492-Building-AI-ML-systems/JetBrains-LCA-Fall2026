import { useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { parse, stringify } from "yaml";
import { api, type LaunchRequest, type TreeNode } from "../api";
import { ProjectTree } from "../components/ProjectTree";
import { Field, Panel } from "../components/ui";
import { useAsync, useMeta } from "../hooks";

type Settings = Record<string, unknown>;

const DEFAULT_SETTINGS: Record<string, Settings> = {
  baseline: {},
  rag: { chunking: "file-aware", top_k: 8, embedding_model: "bge-small-en-v1.5" },
  long_context: { context_window_tokens: 128000, packing: "path-proximity" },
  agent: { max_steps: 12, tools: ["grep", "open_file", "list_dir"] },
};

const MODEL_SUGGESTIONS = [
  "Qwen2.5-Coder-7B-Instruct",
  "DeepSeek-Coder-V2-Lite-Instruct",
  "CodeLlama-13b-Instruct",
  "gpt-4o-mini",
];

const AGENT_TOOLS = ["grep", "open_file", "list_dir", "run_tests"];

function settingsSummary(approach: string, s: Settings): string {
  switch (approach) {
    case "rag":
      return `Chunking: ${s.chunking} • Top-k: ${s.top_k}`;
    case "long_context":
      return `Window: ${Number(s.context_window_tokens) / 1000}k tokens • Packing: ${s.packing}`;
    case "agent":
      return `Max steps: ${s.max_steps} • Tools: ${(s.tools as string[]).join(", ") || "none"}`;
    default:
      return "Official baseline prompt, no extra repository context";
  }
}

export default function ConfigurePage() {
  const meta = useMeta();
  const navigate = useNavigate();
  const tree = useAsync(api.projectTree, []);
  const configs = useAsync(api.configs, []);

  const [task, setTask] = useState(meta.tasks.find((t) => t.id === "bug_localization")?.id ?? meta.tasks[0].id);
  const [approach, setApproach] = useState("rag");
  const [model, setModel] = useState(MODEL_SUGGESTIONS[0]);
  const [repoVersion, setRepoVersion] = useState("long-code-arena@main");
  const [name, setName] = useState("");
  const [settings, setSettings] = useState<Record<string, Settings>>(DEFAULT_SETTINGS);
  const [busy, setBusy] = useState<"launch" | "draft" | null>(null);
  const [notice, setNotice] = useState<{ kind: "ok" | "error"; text: string } | null>(null);
  const [loadedFrom, setLoadedFrom] = useState<string>();

  const current = settings[approach] ?? {};
  const setSetting = (key: string, value: unknown) =>
    setSettings((all) => ({ ...all, [approach]: { ...all[approach], [key]: value } }));

  const request: LaunchRequest = {
    task,
    approach,
    model: model.trim(),
    repo_version: repoVersion.trim() || "main",
    ...(name.trim() ? { name: name.trim() } : {}),
    settings: current,
  };

  const settingsValid =
    approach !== "rag" || (Number(current.top_k) >= 1 && Number(current.top_k) <= 64);
  const ready = Boolean(task && approach && request.model && settingsValid);
  const steps = [
    { label: "Benchmark task", done: Boolean(task) },
    { label: "Context strategy", done: Boolean(approach) },
    { label: "Run settings", done: Boolean(request.model) && settingsValid },
    { label: "Review", done: ready, doneText: "Ready", pendingText: "Incomplete" },
  ];

  const yamlPreview = useMemo(
    () => stringify({ task, approach, model: request.model, repo_version: request.repo_version, settings: current }),
    [task, approach, request.model, request.repo_version, current],
  );

  function loadConfig(node: TreeNode) {
    const file = configs.data?.find((c) => c.path === node.path);
    if (!file) {
      setNotice({ kind: "error", text: `${node.path} is not a YAML config.` });
      return;
    }
    try {
      const doc = parse(file.content) ?? {};
      const docTask = meta.tasks.some((t) => t.id === doc.task) ? doc.task : file.task;
      if (docTask) setTask(docTask);
      const a = doc.approach && meta.approaches.some((x) => x.id === doc.approach) ? doc.approach : approach;
      setApproach(a);
      // Dashboard configs use `model`; the task baselines use model_name (optionally under backbone).
      const docModel = doc.model ?? doc.model_name ?? doc.backbone?.model_name ?? doc.name;
      if (typeof docModel === "string" || typeof docModel === "number") setModel(String(docModel));
      if (doc.repo_version) setRepoVersion(String(doc.repo_version));
      if (doc.settings && typeof doc.settings === "object") {
        setSettings((all) => ({ ...all, [a]: { ...DEFAULT_SETTINGS[a], ...doc.settings } }));
      }
      setLoadedFrom(file.path);
      setNotice({ kind: "ok", text: `Loaded ${file.path}` });
    } catch (err) {
      setNotice({ kind: "error", text: `Couldn’t parse ${file.path}: ${(err as Error).message}` });
    }
  }

  async function launch() {
    setBusy("launch");
    setNotice(null);
    try {
      const run = await api.launch(request);
      navigate(`/runs/${encodeURIComponent(run.id)}`);
    } catch (err) {
      setNotice({ kind: "error", text: (err as Error).message });
    } finally {
      setBusy(null);
    }
  }

  async function saveDraft() {
    setBusy("draft");
    setNotice(null);
    try {
      const file = await api.saveDraft(request);
      setNotice({ kind: "ok", text: `Saved draft to ${file.path}` });
      configs.reload();
      tree.reload();
    } catch (err) {
      setNotice({ kind: "error", text: (err as Error).message });
    } finally {
      setBusy(null);
    }
  }

  return (
    <div className="grid-configure">
      <Panel title="Experiment setup" className="sidebar">
        <ol className="checklist">
          {steps.map((s, i) => (
            <li key={s.label} className={s.done ? "done" : ""}>
              <span className="step-num">{i + 1}</span>
              <div>
                <div className="step-label">{s.label}</div>
                <div className="step-state">{s.done ? s.doneText ?? "Selected" : s.pendingText ?? "Required"}</div>
              </div>
            </li>
          ))}
        </ol>
        <div className="section-label">PROJECT</div>
        {tree.data ? (
          tree.data.length ? (
            <ProjectTree nodes={tree.data} onSelect={loadConfig} selected={loadedFrom} />
          ) : (
            <div className="muted small">Repository folder is empty.</div>
          )
        ) : (
          <div className="muted small">{tree.error ? tree.error.message : "Loading…"}</div>
        )}
        <p className="muted small tree-hint">
          The repo&apos;s task configs. Click one to load its task and model into the form.
        </p>
      </Panel>

      <Panel
        title="Create a controlled evaluation run"
        subtitle="Every setting is saved with the result so comparisons stay fair and reproducible."
        className="main-panel"
      >
        <form
          className="form-grid"
          onSubmit={(e) => {
            e.preventDefault();
            if (ready) launch();
          }}
        >
          <Field label="Benchmark task" htmlFor="task" hint={`Metric: ${meta.tasks.find((t) => t.id === task)?.metric}`}>
            <select id="task" value={task} onChange={(e) => setTask(e.target.value)}>
              {meta.tasks.map((t) => (
                <option key={t.id} value={t.id}>
                  {t.label}
                </option>
              ))}
            </select>
          </Field>
          <Field
            label="Context strategy"
            htmlFor="approach"
            hint={meta.approaches.find((a) => a.id === approach)?.description}
          >
            <select id="approach" value={approach} onChange={(e) => setApproach(e.target.value)}>
              {meta.approaches.map((a) => (
                <option key={a.id} value={a.id}>
                  {a.id === "rag" ? "RAG - repository retrieval" : a.label}
                </option>
              ))}
            </select>
          </Field>
          <Field label="Repository version" htmlFor="repo">
            <input id="repo" value={repoVersion} onChange={(e) => setRepoVersion(e.target.value)} />
          </Field>
          <Field label="Candidate model" htmlFor="model">
            <input
              id="model"
              list="model-suggestions"
              value={model}
              onChange={(e) => setModel(e.target.value)}
              placeholder="Model name or path"
              required
            />
            <datalist id="model-suggestions">
              {MODEL_SUGGESTIONS.map((m) => (
                <option key={m} value={m} />
              ))}
            </datalist>
          </Field>

          <fieldset className="field-wide settings-box">
            <legend>{approach === "rag" ? "Retrieval settings" : approach === "baseline" ? "Baseline settings" : "Strategy settings"}</legend>
            <div className="settings-summary">{settingsSummary(approach, current)}</div>
            {approach === "rag" && (
              <div className="settings-row">
                <label>
                  Chunking
                  <select value={String(current.chunking)} onChange={(e) => setSetting("chunking", e.target.value)}>
                    <option value="file-aware">file-aware</option>
                    <option value="function-level">function-level</option>
                    <option value="fixed-512">fixed 512 tokens</option>
                  </select>
                </label>
                <label>
                  Top-k
                  <input
                    type="number"
                    min={1}
                    max={64}
                    value={Number(current.top_k)}
                    onChange={(e) => setSetting("top_k", e.target.valueAsNumber)}
                  />
                </label>
                <label>
                  Embedding model
                  <input
                    value={String(current.embedding_model ?? "")}
                    onChange={(e) => setSetting("embedding_model", e.target.value)}
                  />
                </label>
              </div>
            )}
            {approach === "long_context" && (
              <div className="settings-row">
                <label>
                  Context window
                  <select
                    value={Number(current.context_window_tokens)}
                    onChange={(e) => setSetting("context_window_tokens", Number(e.target.value))}
                  >
                    {[32000, 64000, 128000, 200000, 1000000].map((n) => (
                      <option key={n} value={n}>
                        {n >= 1e6 ? "1M" : `${n / 1000}k`} tokens
                      </option>
                    ))}
                  </select>
                </label>
                <label>
                  Packing order
                  <select value={String(current.packing)} onChange={(e) => setSetting("packing", e.target.value)}>
                    <option value="path-proximity">path proximity</option>
                    <option value="bm25">BM25 ranked</option>
                    <option value="random">random</option>
                  </select>
                </label>
              </div>
            )}
            {approach === "agent" && (
              <div className="settings-row">
                <label>
                  Max steps
                  <input
                    type="number"
                    min={1}
                    max={100}
                    value={Number(current.max_steps)}
                    onChange={(e) => setSetting("max_steps", e.target.valueAsNumber)}
                  />
                </label>
                <div className="checkbox-group" role="group" aria-label="Tools">
                  <span>Tools</span>
                  {AGENT_TOOLS.map((tool) => {
                    const tools = (current.tools as string[]) ?? [];
                    return (
                      <label key={tool} className="checkbox">
                        <input
                          type="checkbox"
                          checked={tools.includes(tool)}
                          onChange={(e) =>
                            setSetting("tools", e.target.checked ? [...tools, tool] : tools.filter((t) => t !== tool))
                          }
                        />
                        {tool}
                      </label>
                    );
                  })}
                </div>
              </div>
            )}
          </fieldset>

          <Field label="Run name (optional)" htmlFor="name" wide>
            <input
              id="name"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="e.g. E2 – top-k sweep, k=16"
            />
          </Field>

          <div className="field-wide note">
            <div>Same benchmark instances and evaluator will be used for every approach.</div>
            <div className="note-sub">
              Configuration, repository version, runtime, token use, and cost will be recorded.
              {!meta.runner_configured && " No runner is configured on the server yet, so launched runs will stay queued."}
            </div>
          </div>

          <details className="field-wide yaml-preview">
            <summary>config.yaml preview</summary>
            <pre>{yamlPreview}</pre>
          </details>

          <div className="field-wide actions">
            <button type="submit" className="btn btn-primary" disabled={!ready || busy !== null}>
              {busy === "launch" ? "Launching…" : "Launch evaluation"}
            </button>
            <button type="button" className="btn" onClick={saveDraft} disabled={!ready || busy !== null}>
              {busy === "draft" ? "Saving…" : "Save draft"}
            </button>
            {notice && (
              <span className={notice.kind === "ok" ? "notice-ok" : "notice-error"} role="status">
                {notice.text}
              </span>
            )}
          </div>
        </form>
      </Panel>
    </div>
  );
}
