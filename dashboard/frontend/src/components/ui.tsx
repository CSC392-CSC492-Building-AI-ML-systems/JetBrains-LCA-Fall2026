import type { ReactNode } from "react";
import { statusLabel } from "../format";

export function Panel({ title, subtitle, actions, className = "", children }: {
  title?: ReactNode;
  subtitle?: ReactNode;
  actions?: ReactNode;
  className?: string;
  children: ReactNode;
}) {
  return (
    <section className={`panel ${className}`}>
      {(title || actions) && (
        <div className="panel-head">
          <div>
            {title && <h2 className="panel-title">{title}</h2>}
            {subtitle && <p className="panel-subtitle">{subtitle}</p>}
          </div>
          {actions && <div className="panel-actions">{actions}</div>}
        </div>
      )}
      {children}
    </section>
  );
}

export function StatusDot({ status }: { status: string }) {
  return <span className={`dot dot-${status}`} aria-hidden />;
}

/** Status pill: color plus a text label, never color alone. */
export function StatusPill({ status }: { status: string }) {
  return (
    <span className={`pill pill-${status}`}>
      <StatusDot status={status} />
      {statusLabel[status] ?? status}
    </span>
  );
}

export function Field({ label, htmlFor, hint, children, wide = false }: {
  label: string;
  htmlFor: string;
  hint?: string;
  children: ReactNode;
  wide?: boolean;
}) {
  return (
    <div className={`field${wide ? " field-wide" : ""}`}>
      <label htmlFor={htmlFor}>{label}</label>
      {children}
      {hint && <span className="field-hint">{hint}</span>}
    </div>
  );
}

export function Loading({ label = "Loading" }: { label?: string }) {
  return <div className="muted pad">{label}…</div>;
}

export function ErrorNote({ error, hint }: { error: Error; hint?: string }) {
  return (
    <div className="error-note" role="alert">
      <strong>Couldn’t load data.</strong> {error.message}
      {hint && <div className="muted">{hint}</div>}
    </div>
  );
}

export function Empty({ children }: { children: ReactNode }) {
  return <div className="empty">{children}</div>;
}
