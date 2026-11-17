'use client';

export interface LegendClass {
  label: string;
  color: string;
  value?: string;
}

interface MapLegendProps {
  title: string;
  subtitle?: string | null;
  gradient?: string;
  minLabel?: string;
  maxLabel?: string;
  classes?: LegendClass[];
}

/** Positioned below the draw controls, never over them. */
export default function MapLegend({
  title,
  subtitle,
  gradient,
  minLabel,
  maxLabel,
  classes,
}: MapLegendProps) {
  return (
    <div className="pointer-events-auto absolute bottom-16 left-4 z-[1000] max-h-[46%] w-[15rem] overflow-auto rounded-lg border border-rule bg-void/92 px-3 py-2.5 text-[0.75rem] leading-relaxed text-ink-2 shadow-lg backdrop-blur">
      <p className="label mb-0.5 text-ink-3">{title}</p>
      {subtitle && <p className="mb-1.5 text-[0.6875rem] leading-snug text-ink-3">{subtitle}</p>}
      {gradient && (
        <>
          <span className="block h-2 w-full rounded-full" style={{ background: gradient }} />
          <span className="mt-0.5 flex justify-between text-[0.625rem] text-ink-3">
            <span>{minLabel}</span>
            <span>{maxLabel}</span>
          </span>
        </>
      )}
      {classes && classes.length > 0 && (
        <ul className="grid grid-cols-2 gap-x-3 gap-y-0.5">
          {classes.map((entry) => (
            <li key={entry.label} className="flex min-w-0 items-center gap-1.5">
              <span
                aria-hidden
                className="h-2 w-2 shrink-0 rounded-sm border border-rule"
                style={{ background: entry.color }}
              />
              <span className="truncate">{entry.label}</span>
              {entry.value && <span className="ml-auto shrink-0 text-[0.625rem] text-ink-3">{entry.value}</span>}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
