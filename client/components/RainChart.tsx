"use client";

/* Monthly precipitation against its climatological normal.

   Drawn as inline SVG rather than a chart library: one series of discrete monthly
   totals plus one reference line does not justify a dependency, and the project
   rule is that every figure is set in tabular monospaced digits.

   The reading is the point. Bars are the observed month, the stepped rule is the
   1991-2020 normal for that calendar month, and anything breaching the drought
   threshold is picked out in the caution colour. A reader should be able to see
   "this year has been dry" without reading a single label.
*/

import { useMemo, useState } from "react";

export interface RainMonth {
  month: string;
  precip_mm: number;
  normal_mm?: number | null;
  anomaly_pct?: number | null;
  suspect?: string;
}

interface RainChartProps {
  series: RainMonth[];
  normalByMonth?: Record<string, number>;
  /** A month at or below this percentage of normal is picked out. */
  highlightBelow?: number;
}

const W = 960;
const H = 200;
const PAD = { top: 12, right: 74, bottom: 26, left: 44 };

function niceMax(value: number): number {
  if (value <= 25) return 25;
  if (value <= 50) return 50;
  if (value <= 100) return 100;
  if (value <= 200) return 200;
  if (value <= 400) return 400;
  return Math.ceil(value / 200) * 200;
}

export default function RainChart({
  series,
  normalByMonth,
  highlightBelow = -50,
}: RainChartProps) {
  const [hover, setHover] = useState<number | null>(null);

  const model = useMemo(() => {
    if (!series.length) return null;
    const values = series.map((r) => r.precip_mm ?? 0);
    const normals = series.map((r) => {
      const month = r.month?.slice(5, 7);
      if (r.normal_mm != null) return r.normal_mm;
      if (normalByMonth && month) return normalByMonth[month] ?? null;
      return null;
    });
    const peak = Math.max(
      ...values,
      ...normals.map((n) => (n == null ? 0 : n)),
      1,
    );
    const top = niceMax(peak);
    const plotW = W - PAD.left - PAD.right;
    const plotH = H - PAD.top - PAD.bottom;
    const slot = plotW / series.length;
    const barW = Math.max(1, Math.min(slot * 0.74, 14));
    const y = (mm: number) => PAD.top + plotH - (mm / top) * plotH;

    // A stepped reference, one flat step per calendar month so the comparison is
    // month-against-month rather than a misleading interpolation.
    const steps: string[] = [];
    let openMonth: string | null = null;
    series.forEach((row, index) => {
      const value = normals[index];
      const month = row.month?.slice(5, 7) ?? null;
      const x0 = PAD.left + index * slot;
      const x1 = x0 + slot;
      if (value == null) return;
      if (openMonth !== month) {
        if (steps.length) steps.push(`H ${x1.toFixed(1)}`);
        steps.push(`M ${x0.toFixed(1)} ${y(value).toFixed(1)}`);
        openMonth = month;
      }
    });
    if (steps.length) {
      steps.push(
        `H ${(PAD.left + series.length * slot).toFixed(1)}`,
      );
    }

    const ticks = [0, top / 2, top];
    return {
      top,
      plotW,
      plotH,
      slot,
      barW,
      y,
      bars: series.map((row, index) => {
        const value = row.precip_mm ?? 0;
        const anomaly = row.anomaly_pct;
        return {
          index,
          month: row.month,
          value,
          anomaly,
          dry: anomaly != null && anomaly <= highlightBelow,
          suspect: Boolean(row.suspect),
          x: PAD.left + index * slot,
          top: y(value),
          height: Math.max(value > 0 ? 1 : 0, y(0) - y(value)),
        };
      }),
      reference: steps.join(" "),
      ticks,
    };
  }, [series, normalByMonth, highlightBelow]);

  if (!model) return null;

  const active = hover == null ? null : model.bars[hover];
  const shown = active ?? model.bars[model.bars.length - 1];

  return (
    <figure className="fig mt-4">
      <svg
        viewBox={`0 0 ${W} ${H}`}
        className="w-full"
        role="img"
        aria-label="Monthly rainfall against the 1991-2020 normal"
        onMouseLeave={() => setHover(null)}
      >
        {/* axis: the minimum needed to read magnitude, labelled directly */}
        {model.ticks.map((t) => (
          <g key={t}>
            <line
              x1={PAD.left}
              x2={W - PAD.right}
              y1={model.y(t)}
              y2={model.y(t)}
              stroke="var(--rule)"
              strokeWidth={1}
            />
            <text
              x={PAD.left - 6}
              y={model.y(t) + 3}
              textAnchor="end"
              className="fill-[var(--ink-3)] text-[9px]"
            >
              {Math.round(t)}
            </text>
          </g>
        ))}
        <text
          x={PAD.left - 6}
          y={PAD.top - 2}
          textAnchor="end"
          className="fill-[var(--ink-3)] text-[9px]"
        >
          mm
        </text>

        {/* observed months */}
        {model.bars.map((bar) => (
          <rect
            key={bar.index}
            x={bar.x + (model.slot - model.barW) / 2}
            y={bar.top}
            width={model.barW}
            height={bar.height}
            fill={bar.dry ? "var(--caution)" : "var(--ink-2)"}
            opacity={bar.dry ? 1 : 0.75}
            onMouseEnter={() => setHover(bar.index)}
          />
        ))}

        {/* the normal, direct-labelled rather than given a legend box */}
        {model.reference && (
          <>
            <path
              d={model.reference}
              fill="none"
              stroke="var(--ink)"
              strokeWidth={1.25}
            />
            <text
              x={W - PAD.right + 6}
              y={model.y(Math.max(...model.bars.map((b) => b.value))) - 6}
              className="fill-[var(--ink)] text-[9px]"
            >
              1991–
            </text>
            <text
              x={W - PAD.right + 6}
              y={model.y(Math.max(...model.bars.map((b) => b.value))) + 5}
              className="fill-[var(--ink)] text-[9px]"
            >
              2020
            </text>
          </>
        )}

        {/* year ticks, thinned so they never collide */}
        {model.bars.map((bar, i) => {
          const year = bar.month?.slice(0, 4);
          const next = model.bars[i + 1]?.month?.slice(0, 4);
          if (year === next) return null;
          const everyOther = i % 2 === 1 && model.bars[i + 2]?.month?.slice(0, 4) === model.bars[i + 1]?.month?.slice(0, 4);
          if (everyOther) return null;
          return (
            <text
              key={`t${i}`}
              x={bar.x}
              y={H - 8}
              textAnchor="middle"
              className="fill-[var(--ink-3)] text-[9px]"
            >
              {year}
            </text>
          );
        })}

        {/* the read-out, so no hover is required to know where you are */}
        {shown && (
          <text
            x={PAD.left}
            y={H - 8}
            className="fill-[var(--ink-2)] text-[9px]"
          >
            {`${shown.month}  ${Math.round(shown.value)} mm${
              shown.anomaly == null ? '' : `  ${shown.anomaly > 0 ? '+' : ''}${Math.round(shown.anomaly)}%`
            }${shown.dry ? '  dry' : ''}`}
          </text>
        )}
      </svg>
      <figcaption className="mt-1 flex flex-wrap gap-x-4 gap-y-1 text-[10px] text-[var(--ink-3)]">
        <span>Bars: monthly rainfall, mm.</span>
        <span>Stepped line: that calendar month&rsquo;s 1991&ndash;2020 normal.</span>
        <span style={{ color: "var(--caution)" }}>
          Caution marks a month at or below {highlightBelow}% of normal.
        </span>
      </figcaption>
    </figure>
  );
}
