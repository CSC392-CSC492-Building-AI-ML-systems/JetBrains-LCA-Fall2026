import { Navigate, Route, Routes, useLocation } from "react-router-dom";
import { api } from "./api";
import { Header } from "./components/Header";
import { ErrorNote, Loading } from "./components/ui";
import { MetaContext, useAsync } from "./hooks";
import ComparePage from "./pages/ComparePage";
import ConfigurePage from "./pages/ConfigurePage";
import FailuresPage from "./pages/FailuresPage";
import RunsPage from "./pages/RunsPage";

const SCREENS: Record<string, { title: string; subtitle: string }> = {
  configure: { title: "Configure experiment", subtitle: "Create a reproducible evaluation run" },
  runs: { title: "Evaluation run", subtitle: "Track progress and inspect the evidence used by the model" },
  compare: { title: "Compare results", subtitle: "Review performance, cost, latency, and representative failures" },
  failures: { title: "Failure analysis", subtitle: "Separate retrieval failures from reasoning failures" },
};

export default function App() {
  const meta = useAsync(api.meta, []);
  const section = useLocation().pathname.split("/")[1] || "configure";
  const screen = SCREENS[section] ?? SCREENS.configure;

  return (
    <div className="app">
      <Header title={screen.title} subtitle={screen.subtitle} />
      <main className="workspace">
        {meta.error ? (
          <ErrorNote error={meta.error} hint="Is the API running? Start it with: uvicorn app.main:app --reload" />
        ) : !meta.data ? (
          <Loading />
        ) : (
          <MetaContext.Provider value={meta.data}>
            <Routes>
              <Route path="/" element={<Navigate to="/configure" replace />} />
              <Route path="/configure" element={<ConfigurePage />} />
              <Route path="/runs" element={<RunsPage />} />
              <Route path="/runs/:runId" element={<RunsPage />} />
              <Route path="/compare" element={<ComparePage />} />
              <Route path="/failures" element={<FailuresPage />} />
              <Route path="*" element={<Navigate to="/configure" replace />} />
            </Routes>
          </MetaContext.Provider>
        )}
      </main>
    </div>
  );
}
