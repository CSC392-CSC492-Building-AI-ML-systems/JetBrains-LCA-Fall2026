import { createContext, useCallback, useContext, useEffect, useRef, useState } from "react";
import type { Meta } from "./api";

export interface AsyncState<T> {
  data: T | undefined;
  error: Error | undefined;
  loading: boolean;
  reload: () => void;
}

/** Run `load` whenever `deps` change; optionally re-run every `pollMs` while `poll` is true. */
export function useAsync<T>(
  load: () => Promise<T>,
  deps: unknown[],
  { pollMs, poll = false }: { pollMs?: number; poll?: boolean } = {},
): AsyncState<T> {
  const [data, setData] = useState<T>();
  const [error, setError] = useState<Error>();
  const [loading, setLoading] = useState(true);
  const [tick, setTick] = useState(0);
  const loadRef = useRef(load);
  loadRef.current = load;

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    loadRef
      .current()
      .then((value) => {
        if (!cancelled) {
          setData(value);
          setError(undefined);
        }
      })
      .catch((err: Error) => !cancelled && setError(err))
      .finally(() => !cancelled && setLoading(false));
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [...deps, tick]);

  useEffect(() => {
    if (!poll || !pollMs) return;
    const id = window.setInterval(() => setTick((t) => t + 1), pollMs);
    return () => window.clearInterval(id);
  }, [poll, pollMs]);

  const reload = useCallback(() => setTick((t) => t + 1), []);
  return { data, error, loading, reload };
}

export const MetaContext = createContext<Meta | undefined>(undefined);

export function useMeta(): Meta {
  const meta = useContext(MetaContext);
  if (!meta) throw new Error("useMeta outside MetaContext");
  return meta;
}
