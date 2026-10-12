import { NavLink } from "react-router-dom";

const TABS = [
  { to: "/configure", label: "Configure" },
  { to: "/runs", label: "Runs" },
  { to: "/compare", label: "Compare" },
  { to: "/failures", label: "Failures" },
];

export function Header({ title, subtitle }: { title: string; subtitle: string }) {
  return (
    <header className="header">
      <div className="header-titles">
        <div className="brand">LONG CODE ARENA</div>
        <div className="header-row">
          <h1>{title}</h1>
          <span className="header-subtitle">{subtitle}</span>
        </div>
      </div>
      <nav className="tabs" aria-label="Sections">
        {TABS.map((t) => (
          <NavLink key={t.to} to={t.to} className={({ isActive }) => `tab${isActive ? " active" : ""}`}>
            {t.label}
          </NavLink>
        ))}
      </nav>
    </header>
  );
}
