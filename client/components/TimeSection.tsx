"use client";

import { useMemo, useState, type ReactNode } from "react";
import { LineChart as LineIcon, TrendingUp, Loader2 } from "lucide-react";

/* ---------------------------------------------------------------------------
   The time section.

   The previous charts were two panels stacked inside a column that was itself
   inside a grid, which put each plot at roughly 480x150 rendered pixels with 9px
   type. That is not a chart you read, it is a chart you lean into. The
   Digital Earth Africa conservancies dashboard, which the first users responded
   to, gave every question its own full-width plot with type you could read across
   a room — and that is the standard this section is held to.

   Three decisions follow from that reference:

   One question, one plot. Rainfall totals, rainfall anomaly and vegetation
   condition are three different claims about three different units, and
   overlaying them in one frame forced a shared axis that flattened all three.
   They are separate here, full width, in the order a reader asks them.

   The range follows the question. A user who picked a one-year window was
   looking at a snapshot and was getting 35 years of rainfall back, which reads
   as the product answering a different question than the one asked. The range
   control defaults to the analysed window and offers the full record as an
   explicit, labelled choice for someone who came for the trend.

   The vegetation series is a headline, not a footnote. It used to sit behind a
   "Build a vegetation series" ghost button that only appeared when the series
   was missing, which is the least discoverable place to put the thing the first
   audience responded to most strongly. It is offered here, by name, with its
   cost stated.
   --------------------------------------------------------------------------- */

export type RainPoint = {
  month: string;
  precip_mm?: number | null;
  normal_mm?: number | null;
  anomaly_pct?: number | null;
  suspect?: boolean;
};

export type VegPoint = {
  month: string;
  value: number | null;
  min?: number | null;
  max?: number | null;
  normal?: number | null;
  anomaly?: number | null;
  anomaly_pct?: number | null;
};

type Range = "window" | "record";

/* --- geometry ---------------------------------------------------------------
   One viewBox width for all three plots so the x axis, the year ticks and the
   hover crosshair land in exactly the same pixels in every chart. A reader who
   has learned where 2019 is in one plot has learned it in all three. */
const W = 960;
const PLOT_H = 268;
const AXIS_H = 30;
const PAD = { top: 16, right: 18, bottom: 0, left: 62 };
const H = PLOT_H + AXIS_H;
const INNER_W = W - PAD.left - PAD.right;

const YEAR_RE = /^(\d{4})-/;
const monthIndex = (month: string) => {
  const y = Number(month.slice(0, 4));
  const m = Number(month.slice(5, 7));
  return y * 12 + (m - 1);
};
const shortLabel = (month: string) => `${month.slice(5, 7)}/${month.slice(2, 4)}`;

/* Axis ticks on round numbers.

   A scale that ends at 251 mm because 251 is what the data needed makes a chart
   look computed rather than read. This walks the raw step up to the nearest
   1 / 2 / 2.5 / 5 x 10^n and snaps the bounds to it, so an axis always reads
   0 / 50 / 100 / 150 / 200 / 250 mm. It is the cheapest possible difference
   between a chart you trust and a chart you squint at. */
function niceScale(lo: number, hi: number, target = 5) {
  if (!(hi > lo)) return { ticks: [lo, lo + 1], bottom: lo, top: lo + 1 };
  const raw = (hi - lo) / target;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const n = raw / mag;
  const step = (n <= 1 ? 1 : n <= 2 ? 2 : n <= 2.5 ? 2.5 : n <= 5 ? 5 : 10) * mag;
  const bottom = Math.floor(lo / step) * step;
  const top = Math.ceil(hi / step) * step;
  const ticks: number[] = [];
  // Built by index rather than by repeated addition, which drifts on steps like
  // 2.5 and lands the last gridline a hair above the plot.
  for (let i = 0; bottom + i * step <= top + step * 1e-6; i++) {
    ticks.push(Number((bottom + i * step).toPrecision(12)));
  }
  return { ticks, bottom, top };
}

/* A ramp that means the same thing on every plot: the value is already on an
   axis, so hue is never the only carrier — it is the pre-attentive path to the
   same information. */
const tone = (pct: number | null | undefined, severeAt: number) => {
  if (pct == null) return "var(--text-3)";
  if (pct <= -severeAt) return "var(--bare)";
  if (pct <= -10) return "var(--stressed)";
  if (pct >= 10) return "var(--deep)";
  return "var(--text-3)";
};

/* --- shared axis furniture -------------------------------------------------- */

/* Year ticks, thinned so they never collide and never sit under a bar's label. */
function useYearTicks(months: string[], innerW: number) {
  return useMemo(() => {
    const out: Array<{ year: string; x: number; index: number }> = [];
    let last = "";
    months.forEach((m, i) => {
      const y = m.match(YEAR_RE)?.[1];
      if (!y || y === last) return;
      last = y;
      out.push({ year: y, x: PAD.left + (i / Math.max(1, months.length - 1)) * innerW, index: i });
    });
    // Below a certain density, only every other year gets a label.
    const minGap = 46;
    const kept = out.filter((t, i) => i === 0 || t.x - out[i - 1].x >= minGap);
    return { all: out, kept };
  }, [months, innerW]);
}

function Gridline({ x1, x2, y, label }: { x1: number; x2: number; y: number; label?: string }) {
  return (
    <g>
      <line x1={x1} x2={x2} y1={y} y2={y} stroke="var(--line)" strokeWidth={1} />
      {label != null && (
        <text
          x={x1 - 12}
          y={y + 4}
          textAnchor="end"
          className="fill-[var(--text-3)]"
          style={{ fontSize: 12 }}
        >
          {label}
        </text>
      )}
    </g>
  );
}

function YearAxis({ months, innerW }: { months: string[]; innerW: number }) {
  const { all, kept } = useYearTicks(months, innerW);
  return (
    <g transform={`translate(0, ${PLOT_H + 20})`}>
      {all.map((t) => (
        <line
          key={`tick-${t.year}`}
          x1={t.x}
          x2={t.x}
          y1={-PLOT_H - 14}
          y2={-6}
          stroke="var(--line)"
          strokeWidth={1}
        />
      ))}
      {kept.map((t) => (
        <text
          key={`lbl-${t.year}`}
          x={t.x}
          y={4}
          textAnchor="middle"
          className="fill-[var(--text-2)]"
          style={{ fontSize: 12, fontWeight: 500 }}
        >
          {t.year}
        </text>
      ))}
    </g>
  );
}

function Crosshair({ x, label }: { x: number | null; label: string }) {
  if (x == null) return null;
  return (
    <g>
      <line x1={x} x2={x} y1={0} y2={PLOT_H} stroke="var(--line-2)" strokeWidth={1} />
      <circle cx={x} cy={0} r={0} />
      <text
        x={Math.min(Math.max(x, PAD.left + 46), W - PAD.right - 46)}
        y={-8}
        textAnchor="middle"
        className="fill-[var(--text)]"
        style={{ fontSize: 12.5, fontWeight: 600 }}
      >
        {label}
      </text>
    </g>
  );
}

/* A plot is a frame, not a component: each chart owns its marks because their
   shapes are genuinely different, and the shared parts are the axis furniture
   above. */
function Frame({
  months,
  children,
  onMove,
}: {
  months: string[];
  children: ReactNode;
  onMove?: (index: number | null) => void;
}) {
  const x = (i: number) =>
    PAD.left + (i / Math.max(1, months.length - 1)) * INNER_W;
  return (
    <figure className="m-0">
      <div className="overflow-x-auto">
        <svg
          viewBox={`0 0 ${W} ${H + PAD.top}`}
          className="w-full min-w-[760px]"
          role="img"
          onMouseLeave={() => onMove?.(null)}
        >
          <g transform={`translate(0, ${PAD.top})`}>
            {children}
            <YearAxis months={months} innerW={INNER_W} />
          </g>
          {/* The hit area. One transparent rect over the plot means the reader
              does not have to find a 4px bar to get a reading. */}
          <rect
            x={PAD.left}
            y={PAD.top}
            width={INNER_W}
            height={PLOT_H}
            fill="transparent"
            onMouseMove={(e) => {
              if (!onMove) return;
              const box = (e.target as SVGRectElement).ownerSVGElement?.getBoundingClientRect();
              if (!box) return;
              const rel = ((e.clientX - box.left) / box.width) * W;
              const t = (rel - PAD.left) / INNER_W;
              onMove(Math.round(Math.max(0, Math.min(1, t)) * (months.length - 1)));
            }}
          />
        </svg>
      </div>
      <span className="sr-only">{x(0)}</span>
    </figure>
  );
}

/* --- the read-out -----------------------------------------------------------
   The single biggest change from the old charts. A reading is not a thing you
   chase with a cursor, it is a thing that is simply always written down, large,
   in the corner of its own plot. Hover adds detail; it is not the only way to
   find out what the chart says. */
function Readout({
  month,
  primary,
  secondary,
}: {
  month: string | null;
  primary: string;
  secondary?: string;
}) {
  return (
    <div className="flex flex-wrap items-baseline gap-x-4 gap-y-1">
      <span className="fig text-[0.9375rem] font-semibold tracking-[-0.01em] text-signal">
        {month ?? "—"}
      </span>
      <span className="fig text-[0.9375rem] font-medium text-ink">{primary}</span>
      {secondary && <span className="fig text-[0.8125rem] text-ink-3">{secondary}</span>}
    </div>
  );
}

/* --- chart 1: rainfall totals ----------------------------------------------- */

function RainChart({ data }: { data: RainPoint[] }) {
  const [hover, setHover] = useState<number | null>(null);
  const months = data.map((d) => d.month);

  const { y, ticks } = useMemo(() => {
    const scale = niceScale(0, Math.max(20, ...data.flatMap((d) => [d.precip_mm ?? 0, d.normal_mm ?? 0])) * 1.04);
    return {
      y: (v: number) => PLOT_H - (v / scale.top) * PLOT_H,
      ticks: scale.ticks,
    };
  }, [data]);

  const slot = INNER_W / Math.max(1, data.length);
  const barW = Math.max(1.5, Math.min(slot * 0.68, 13));
  const at = (i: number) => PAD.left + i * slot + (slot - barW) / 2;

  // The normal is a step, not a line: it is a single value repeated for twelve
  // months, and drawing it as a smooth line would invent variation that is not
  // in the data.
  const steps = useMemo(() => {
    const out: string[] = [];
    let currentYear = "";
    let started = false;
    data.forEach((d, i) => {
      const year = d.month.slice(0, 4);
      if (year !== currentYear) {
        if (started) out.push(`H ${PAD.left + i * slot + slot / 2}`);
        currentYear = year;
        out.push(`M ${PAD.left + i * slot} ${y(d.normal_mm ?? 0)}`);
        out.push(`L ${PAD.left + i * slot + slot} ${y(d.normal_mm ?? 0)}`);
        started = true;
      } else {
        out.push(`L ${PAD.left + (i + 1) * slot} ${y(d.normal_mm ?? 0)}`);
      }
    });
    return out.join(" ");
  }, [data, y, slot]);

  const active = hover != null ? data[hover] : data[data.length - 1];

  return (
    <section className="chart">
      <header className="chart-head">
        <div>
          <h3 className="chart-title">Rainfall</h3>
          <p className="chart-sub">Monthly total, against the 1991–2020 normal for that month</p>
        </div>
        <Readout
          month={active?.month ?? null}
          primary={active ? `${Math.round(active.precip_mm ?? 0)} mm` : "—"}
          secondary={
            active?.anomaly_pct != null
              ? `${active.anomaly_pct > 0 ? "+" : ""}${Math.round(active.anomaly_pct)}% vs normal`
              : undefined
          }
        />
      </header>
      <Frame months={months} onMove={setHover}>
        {ticks.map((t, i) => (
          <Gridline
            key={i}
            x1={PAD.left}
            x2={PAD.left + INNER_W}
            y={y(t)}
            label={t === 0 ? "0" : `${formatTick(t)} mm`}
          />
        ))}
        {/* The normal sits behind the bars. Drawn on top it was a grey sawtooth
            cutting across every month, competing with the thing the reader
            actually came for. */}
        <path
          d={steps}
          fill="none"
          stroke="var(--text-3)"
          strokeWidth={1.25}
          opacity={0.7}
        />
        {data.map((d, i) => {
          const v = d.precip_mm ?? 0;
          return (
            <rect
              key={d.month}
              x={at(i)}
              y={y(v)}
              width={barW}
              height={Math.max(v > 0 ? 1 : 0, y(0) - y(v))}
              rx={Math.min(2, barW / 3)}
              fill={tone(d.anomaly_pct, -50)}
              opacity={hover == null || hover === i ? 1 : 0.5}
            />
          );
        })}
        <Crosshair
          x={hover != null ? PAD.left + hover * slot + slot / 2 : null}
          label={hover != null ? shortLabel(data[hover].month) : ""}
        />
      </Frame>
    </section>
  );
}

/* Ticks are already round, so this only has to keep enough decimals to tell
   adjacent ticks apart and then drop the trailing ones. Rounding to one decimal
   collapsed 0.25 and 0.30 both to "0.3" and printed the same label three times
   down the NDVI axis, which has a 0.05 step. */
const formatTick = (v: number) => {
  const fixed = v.toFixed(2);
  return fixed.includes(".") ? fixed.replace(/\.?0+$/, "") : fixed;
};

/* --- chart 2: rainfall anomaly ----------------------------------------------
   Its own plot, not an overlay. The question "was it dry?" and the question "how
   much did it rain?" have different units, and putting a percentage on top of a
   millimetre axis makes both harder to read. */
function AnomalyChart({ data }: { data: RainPoint[] }) {
  const [hover, setHover] = useState<number | null>(null);
  const months = data.map((d) => d.month);

  /* The axis is set from the 97th percentile, not the maximum.

     A single month at +336% — a real value, this catchment has months like that
     — set a symmetric axis that compressed all 114 other months into a hairline
     around zero. The chart was technically correct and practically unreadable,
     which is the worst of both. Outliers are now clamped to the edge and drawn
     with a cap so they still read as "off the scale" rather than as "at the
     maximum". */
  const { y, extent, ticks, clipped } = useMemo(() => {
    const magnitudes = data
      .map((d) => d.anomaly_pct)
      .filter((v): v is number => v != null)
      .map(Math.abs)
      .sort((a, b) => a - b);
    const p97 = magnitudes[Math.min(magnitudes.length - 1, Math.floor(magnitudes.length * 0.97))] ?? 0;
    const scale = niceScale(-Math.max(20, p97 * 1.12), Math.max(20, p97 * 1.12), 4);
    const top = scale.top;
    const mid = PLOT_H / 2;
    const s = (v: number) => mid - (Math.max(-top, Math.min(top, v)) / top) * (PLOT_H / 2);
    return {
      y: s,
      extent: top,
      ticks: scale.ticks,
      clipped: magnitudes.filter((m) => m > top).length,
    };
  }, [data]);

  const slot = INNER_W / Math.max(1, data.length);
  const barW = Math.max(1.5, Math.min(slot * 0.68, 13));

  const active = hover != null ? data[hover] : data[data.length - 1];

  return (
    <section className="chart">
      <header className="chart-head">
        <div>
          <h3 className="chart-title">Departure from normal</h3>
          <p className="chart-sub">
            The same months as a percentage of normal. Zero is normal; below it, dry.
            {clipped > 0 &&
              ` ${clipped} month${clipped === 1 ? " runs past the scale and is" : "s run past the scale and are"} capped at the edge.`}
          </p>
        </div>
        <Readout
          month={active?.month ?? null}
          primary={
            active?.anomaly_pct != null
              ? `${active.anomaly_pct > 0 ? "+" : ""}${Math.round(active.anomaly_pct)}%`
              : "—"
          }
          secondary={
            active?.precip_mm != null
              ? `${Math.round(active.precip_mm)} mm vs ${Math.round(active.normal_mm ?? 0)} mm`
              : undefined
          }
        />
      </header>
      <Frame months={months} onMove={setHover}>
        {ticks.map((t, i) => (
          <Gridline
            key={i}
            x1={PAD.left}
            x2={PAD.left + INNER_W}
            y={y(t)}
            label={`${t > 0 ? "+" : ""}${formatTick(t)}%`}
          />
        ))}
        {data.map((d, i) => {
          const v = d.anomaly_pct;
          if (v == null) return null;
          const top = Math.min(y(v), y(0));
          const capped = Math.abs(v) > extent;
          return (
            <g key={d.month} opacity={hover == null || hover === i ? 1 : 0.5}>
              <rect
                x={PAD.left + i * slot + (slot - barW) / 2}
                y={top}
                width={barW}
                height={Math.max(1.5, Math.abs(y(v) - y(0)))}
                rx={Math.min(2, barW / 3)}
                fill={tone(v, -50)}
              />
              {/* The cap. A flat edge instead of a rounded one reads as "there is
                  more of this above" without needing an arrow. */}
              {capped && (
                <rect
                  x={PAD.left + i * slot + (slot - barW) / 2}
                  y={v > 0 ? top : top + Math.abs(y(v) - y(0)) - 2}
                  width={barW}
                  height={2}
                  fill="var(--void)"
                />
              )}
            </g>
          );
        })}
        <Crosshair
          x={hover != null ? PAD.left + hover * slot + slot / 2 : null}
          label={hover != null ? shortLabel(data[hover].month) : ""}
        />
      </Frame>
    </section>
  );
}

/* --- chart 3: vegetation condition ------------------------------------------
   The one the first audience asked for by name. The area between the monthly
   minimum and maximum is real data the backend already returns and the old chart
   threw away; showing it turns a single line into a statement about how much the
   ground inside one boundary varies, which is the thing a grazier actually wants
   to know. */
function VegChart({ data, source, thinMonths }: { data: VegPoint[]; source?: string | null; thinMonths?: string[] }) {
  const [hover, setHover] = useState<number | null>(null);
  const months = data.map((d) => d.month);

  const { y, ticks } = useMemo(() => {
    /* The axis is set by the mean, over its 1st to 99th percentile.

       Measured on Naibunga Upper — 196 months, 5,596 pixels a month — the area
       mean lives between 0.23 and 0.69. Snapping to a 0.05 grid off those
       percentiles gives 0.20 to 0.65, which is ten gridlines and a seasonal
       signal you can actually see.

       The monthly min/max band was drawn here and then removed. It is real data
       and the backend returns it, but across a landscape it spans almost the
       whole index (-0.30 to 0.997 on that area), so the only way to keep both
       it and a readable axis was to clip it: it ran past the axis in 191 of 196
       months. Clipped to that degree it is two bars pinned at the top and
       bottom edges, which looks like information and carries none. The spread is
       still reported where it belongs — the vegetation module's middle-50% row,
       its range row, and valid-pixel count under Show the working. */
    const means = data
      .flatMap((d) => [d.value, d.normal])
      .filter((v): v is number => v != null)
      .sort((a, b) => a - b);
    const at = (q: number) =>
      means[Math.min(means.length - 1, Math.max(0, Math.round(q * (means.length - 1))))];
    const lo = Math.max(0, Math.floor((at(0.01) - 0.02) * 20) / 20);
    const hi = Math.min(1, Math.ceil((at(0.99) + 0.02) * 20) / 20);
    const s = (v: number) => PLOT_H - ((Math.max(lo, Math.min(hi, v)) - lo) / (hi - lo || 1)) * PLOT_H;
    const out: number[] = [];
    for (let v = lo; v <= hi + 1e-9; v += 0.05) out.push(Number(v.toFixed(2)));
    return { y: s, ticks: out };
  }, [data]);

  // The scale functions live inside the memos that use them rather than beside
  // them, so a change to either can never be picked up by one memo and missed by
  // the other.
  const xOf = (i: number) => PAD.left + (i / Math.max(1, data.length - 1)) * INNER_W;

  const { line, normalLine } = useMemo(() => {
    const x = (i: number) => PAD.left + (i / Math.max(1, data.length - 1)) * INNER_W;
    const line = data
      .map((d, i) => (d.value == null ? null : `${i === 0 ? "M" : "L"} ${x(i)} ${y(d.value)}`))
      .filter(Boolean)
      .join(" ");
    const segs: string[] = [];
    data.forEach((d, i) => {
      if (d.normal == null) return;
      if (i === 0 || data[i - 1]?.normal == null) segs.push(`M ${x(i)} ${y(d.normal)}`);
      else segs.push(`L ${x(i)} ${y(d.normal)}`);
    });
    return { line, normalLine: segs.join(" ") };
  }, [data, y]);

  const active = hover != null ? data[hover] : data[data.length - 1];
  const activeMonth = active?.month;
  const thinSet = useMemo(() => new Set(thinMonths ?? []), [thinMonths]);

  return (
    <section className="chart">
      <header className="chart-head">
        <div>
          <h3 className="chart-title">Vegetation condition</h3>
          <p className="chart-sub">
            Area-mean NDVI for the boundary, against the long-term normal for
            each calendar month
            {source ? ` · ${source}` : ""}
          </p>
        </div>
        <Readout
          month={active?.month ?? null}
          primary={active?.value != null ? active.value.toFixed(3) : "—"}
          secondary={
            active?.anomaly != null
              ? `${active.anomaly > 0 ? "+" : ""}${active.anomaly.toFixed(3)} vs normal`
              : undefined
          }
        />
      </header>
      <Frame months={months} onMove={setHover}>
        {ticks.map((t, i) => (
          <Gridline
            key={i}
            x1={PAD.left}
            x2={PAD.left + INNER_W}
            y={y(t)}
            label={formatTick(t)}
          />
        ))}
        {normalLine && (
          <path
            d={normalLine}
            fill="none"
            stroke="var(--text-2)"
            strokeWidth={1.75}
            strokeDasharray="5 5"
            opacity={0.8}
          />
        )}
        <path
          d={line}
          fill="none"
          stroke="var(--signal)"
          strokeWidth={2.5}
          strokeLinejoin="round"
          strokeLinecap="round"
          className="reveal-line"
        />
        {data.map((d, i) =>
          thinSet.has(d.month) ? (
            <circle key={d.month} cx={xOf(i)} cy={y(d.value ?? 0)} r={3.5} fill="var(--stressed)" />
          ) : null
        )}
        {activeMonth && hover != null && (
          <circle cx={xOf(hover)} cy={y(data[hover].value ?? 0)} r={5} fill="var(--signal)" />
        )}
        <Crosshair
          x={hover != null ? xOf(hover) : null}
          label={hover != null ? shortLabel(data[hover].month) : ""}
        />
      </Frame>
    </section>
  );
}

/* --- the section ------------------------------------------------------------ */

export default function TimeSection({
  rain,
  vegetation,
  windowStart,
  windowEnd,
  vegetationSource,
  thinMonths,
  onBuildVegetation,
  buildingVegetation,
  jobState,
  jobMonths,
  vegetationNotice,
}: {
  rain: RainPoint[];
  vegetation: VegPoint[];
  windowStart: string;
  windowEnd: string;
  vegetationSource?: string | null;
  thinMonths?: string[];
  onBuildVegetation?: () => void;
  buildingVegetation?: boolean;
  jobState?: string | null;
  jobMonths?: number | null;
  vegetationNotice?: string | null;
}) {
  const [range, setRange] = useState<Range>("window");

  // The bounds, computed once rather than inside a predicate, so the two
  // filtering memos below have a stable dependency and do not have to be told
  // about a closure.
  const lo = monthIndex(windowStart.slice(0, 7));
  const hi = monthIndex(windowEnd.slice(0, 7));

  const shownRain = useMemo(
    () =>
      range === "record"
        ? rain
        : rain.filter((d) => {
            const i = monthIndex(d.month);
            return i >= lo && i <= hi;
          }),
    [rain, range, lo, hi]
  );
  const shownVeg = useMemo(
    () =>
      range === "record"
        ? vegetation
        : vegetation.filter((d) => {
            const i = monthIndex(d.month);
            return i >= lo && i <= hi;
          }),
    [vegetation, range, lo, hi]
  );

  const hasRain = shownRain.length > 0;
  const hasVeg = shownVeg.length > 0;
  const hasAny = hasRain || hasVeg;

  /* The three numbers a person actually wants off this section, computed over
     the range on screen rather than over the full record. Changing the range
     changes these, which is the point: they are the section's summary, not a
     caption. Declared before the early return below, because a hook after a
     conditional return is a hook that runs in a different order depending on
     the data. */
  const stats = useMemo(() => {
    const total = shownRain.reduce((s, d) => s + (d.precip_mm ?? 0), 0);
    const normal = shownRain.reduce((s, d) => s + (d.normal_mm ?? 0), 0);
    const dry = shownRain.filter((d) => d.anomaly_pct != null && d.anomaly_pct <= -25).length;
    const first = shownVeg[0]?.value ?? null;
    const last = shownVeg[shownVeg.length - 1]?.value ?? null;
    const vegChange = first != null && last != null ? last - first : null;
    return {
      total,
      totalNormal: normal,
      dry,
      first,
      last,
      vegChange,
    };
  }, [shownRain, shownVeg]);

  if (!hasAny) return null;

  const months = hasRain ? shownRain.map((d) => d.month) : shownVeg.map((d) => d.month);

  return (
    <section className="panel overflow-hidden p-0">
      <header className="flex flex-wrap items-end justify-between gap-x-6 gap-y-4 border-b border-rule px-6 py-5 sm:px-8 sm:py-6">
        <div>
          <p className="label">Over time</p>
          <h2 className="display display-md mt-2">The record</h2>
        </div>
        <div className="flex flex-col items-start gap-2.5 sm:items-end">
          <Segmented
            value={range}
            onChange={setRange}
            options={[
              { value: "window", label: "This window" },
              { value: "record", label: "Full record" },
            ]}
          />
          <p className="fig text-[0.75rem] text-ink-3">
            {range === "window"
              ? `${windowStart.slice(0, 7)} → ${windowEnd.slice(0, 7)} · ${months.length} months`
              : `${months[0]?.slice(0, 7)} → ${months[months.length - 1]?.slice(0, 7)} · ${months.length} months`}
          </p>
        </div>
      </header>

      <div className="px-6 py-2 sm:px-8">
        <div className="grid divide-rule border-b border-rule sm:grid-cols-2 sm:divide-x lg:grid-cols-4">
          <Stat
            label="Rain over this span"
            value={hasRain ? `${Math.round(stats.total).toLocaleString()}` : "—"}
            unit="mm"
            sub={
              hasRain && stats.totalNormal > 0
                ? // One decimal, deliberately. This area is within half a percent
                  // of its long-term normal, and a rounded "0%" reads as a broken
                  // number rather than as the finding that it is.
                  `${(((stats.total - stats.totalNormal) / stats.totalNormal) * 100).toFixed(1)}% vs normal`
                : undefined
            }
          />
          <Stat
            label="Dry months"
            value={hasRain ? String(stats.dry) : "—"}
            sub={hasRain ? "at or below 75% of normal" : undefined}
          />
          <Stat
            label="Vegetation now"
            value={stats.last != null ? stats.last.toFixed(3) : "—"}
            unit="NDVI"
          />
          <Stat
            label="Change across span"
            value={stats.vegChange != null ? `${stats.vegChange > 0 ? "+" : ""}${stats.vegChange.toFixed(3)}` : "—"}
            sub={
              hasVeg
                ? "first month to last"
                : "build the vegetation series to see it"
            }
            tone={stats.vegChange != null ? (stats.vegChange >= 0 ? "up" : "down") : undefined}
          />
        </div>

        {hasRain && (
          <div className="divide-y divide-rule">
            <RainChart data={shownRain} />
            <AnomalyChart data={shownRain} />
          </div>
        )}

        {hasVeg ? (
          <div className="border-t border-rule">
            <VegChart data={shownVeg} source={vegetationSource} thinMonths={thinMonths} />
          </div>
        ) : (
          <div className="border-t border-rule">
            <div className="flex flex-col gap-5 py-10 sm:flex-row sm:items-center sm:justify-between">
              <div className="max-w-[54ch]">
                <div className="flex items-center gap-2.5">
                  {jobState ? (
                    <span className="beat h-3.5 w-3.5 rounded-full bg-signal" />
                  ) : (
                    <TrendingUp className="h-4 w-4 text-signal" />
                  )}
                  <h3 className="chart-title">Vegetation condition over time</h3>
                </div>
                <p className="mt-2.5 text-[0.9375rem] leading-relaxed text-ink-2">
                  The snapshot above is one moment. A monthly series from the same
                  boundary is what shows whether this place is greening or browning —
                  and whether that tracks the rain.
                </p>
                <p className="mt-2 text-[0.8125rem] leading-relaxed text-ink-3">
                  {jobState
                    ? // The button used to disappear the moment the job was queued,
                      // leaving a section that was silently empty. It now says what
                      // is happening, for as long as it is happening, and the wait is
                      // quantified because a spinner over an unquantified wait is the
                      // thing this replaces.
                      `Building now — reading ${jobMonths ? `${jobMonths} months` : "month by month"} from MODIS. This usually takes a few minutes, and you can leave this page open.`
                    : (vegetationNotice ??
                      "Read offline from MODIS, one month at a time. The wait is minutes rather than seconds, and it does not grow with the size of the area.")}
                </p>
              </div>
              {onBuildVegetation && !jobState && (
                <Buttonish busy={buildingVegetation} onClick={onBuildVegetation}>
                  <LineIcon className="h-4 w-4" />
                  {buildingVegetation ? "Building the series" : "Build the series"}
                </Buttonish>
              )}
            </div>
          </div>
        )}
      </div>

      <footer className="border-t border-rule px-6 py-4 sm:px-8">
        <p className="max-w-[76ch] text-[0.75rem] leading-relaxed text-ink-3">
          Rainfall is ERA5, a reanalysis that assimilates observations rather than a
          gauge reading. Vegetation is MODIS MOD13Q1 at 250 m. Co-variation between the
          two is not attribution: vegetation responds to rain with a lag that varies by
          season, and water is not always the limiting factor in semi-arid rangeland.
        </p>
      </footer>
    </section>
  );
}

function Stat({
  label,
  value,
  unit,
  sub,
  tone,
}: {
  label: string;
  value: string;
  unit?: string;
  sub?: string;
  tone?: "up" | "down";
}) {
  return (
    <div className="px-0 py-6 sm:px-6 sm:first:pl-0 lg:px-8">
      <p className="label">{label}</p>
      <p
        className={`fig mt-2.5 flex items-baseline gap-1.5 text-[clamp(1.75rem,3vw,2.375rem)] font-medium leading-none tracking-[-0.03em] ${
          tone === "up" ? "text-deep" : tone === "down" ? "text-stressed" : "text-ink"
        }`}
      >
        {value}
        {unit && <span className="text-[0.9375rem] font-normal tracking-normal text-ink-3">{unit}</span>}
      </p>
      {sub && <p className="mt-2 text-[0.75rem] leading-snug text-ink-3">{sub}</p>}
    </div>
  );
}

function Buttonish({
  busy,
  onClick,
  children,
}: {
  busy?: boolean;
  onClick: () => void;
  children: ReactNode;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={busy}
      className={`btn btn-primary h-11 shrink-0 px-5 text-[0.875rem] ${busy ? "pointer-events-none" : ""}`}
    >
      {busy ? <Loader2 className="h-4 w-4 animate-spin" /> : null}
      {children}
    </button>
  );
}

/* A track with one lit thumb. Measured from the DOM rather than assumed, so the
   thumb lands correctly whatever the label widths turn out to be. */
function Segmented<T extends string>({
  value,
  onChange,
  options,
}: {
  value: T;
  onChange: (v: T) => void;
  options: Array<{ value: T; label: string }>;
}) {
  const [track, setTrack] = useState<HTMLDivElement | null>(null);
  const index = Math.max(0, options.findIndex((o) => o.value === value));
  const activeEl = track?.querySelectorAll<HTMLElement>("[data-seg]")[index];

  return (
    <div
      ref={setTrack}
      className="segmented relative"
      role="tablist"
      onKeyDown={(e) => {
        if (e.key !== "ArrowRight" && e.key !== "ArrowLeft") return;
        e.preventDefault();
        const next =
          e.key === "ArrowRight"
            ? Math.min(options.length - 1, index + 1)
            : Math.max(0, index - 1);
        onChange(options[next].value);
      }}
    >
      {activeEl && (
        <span
          className="segmented-thumb"
          style={{
            width: activeEl.offsetWidth,
            transform: `translateX(${activeEl.offsetLeft - 3}px)`,
          }}
        />
      )}
      {options.map((o) => (
        <button
          key={o.value}
          type="button"
          role="tab"
          data-seg
          aria-selected={value === o.value}
          tabIndex={value === o.value ? 0 : -1}
          onClick={() => onChange(o.value)}
          className="segmented-item"
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}
