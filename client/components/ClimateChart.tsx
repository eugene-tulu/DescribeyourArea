"use client";

/* Rainfall and vegetation over one shared time axis, as two stacked panels.

   Deliberately not a dual-axis chart. Rainfall in millimetres and NDVI in [-1, 1]
   cannot share a scale without implying they are comparable, and this product
   exists to avoid exactly that kind of implied claim. Two panels on a shared
   time axis let the reader compare the *shape* of the seasons vertically, which
   is the honest reading, without putting two units on one ruler.

   Each panel carries its own normal, its own axis label, and its own epistemic
   status, because the two series are not the same kind of evidence: rainfall is
   a reanalysis, vegetation an index derived from a reflectance product.
*/

import { useMemo, useState } from "react";

export interface RainMonth {
  month: string;
  precip_mm: number;
  normal_mm?: number | null;
  anomaly_pct?: number | null;
  suspect?: string;
}

export interface VegMonth {
  month: string;
  value: number;
  normal?: number | null;
  anomaly_pct?: number | null;
}

interface ClimateChartProps {
  rain?: RainMonth[];
  vegetation?: VegMonth[];
  /** Normal basis for each series, e.g. "1991-2020". */
  rainNormal?: string | null;
  vegetationNormal?: string | null;
  vegetationStatus?: string | null;
  vegetationSource?: string | null;
}

const W = 960;
const RAIN_H = 150;
const VEG_H = 110;
const GAP = 26;
const PAD = { top: 10, right: 66, bottom: 24, left: 42 };
const TOTAL_H = PAD.top + RAIN_H + GAP + VEG_H + PAD.bottom;

function niceTop(value: number): number {
  if (value <= 25) return 25;
  if (value <= 60) return 60;
  if (value <= 120) return 120;
  if (value <= 250) return 250;
  if (value <= 500) return 500;
  return Math.ceil(value / 250) * 250;
}

function yearTicks(months: string[]): Array<{ x: number; year: string }> {
  const out: Array<{ x: number; year: string }> = [];
  months.forEach((m, i) => {
    const year = m?.slice(0, 4);
    const next = months[i + 1]?.slice(0, 4);
    if (year && year !== next) out.push({ x: i, year });
  });
  return out;
}

export default function ClimateChart({
  rain = [],
  vegetation = [],
  rainNormal,
  vegetationNormal,
  vegetationStatus,
  vegetationSource,
}: ClimateChartProps) {
  const [hover, setHover] = useState<number | null>(null);

  const model = useMemo(() => {
    const months = (rain.length ? rain.map((r) => r.month) : vegetation.map((v) => v.month))
      .slice();
    if (!months.length) return null;
    const plotW = W - PAD.left - PAD.right;
    const slot = plotW / months.length;

    const rainTop = niceTop(
      Math.max(
        ...rain.map((r) => r.precip_mm ?? 0),
        ...rain.map((r) => r.normal_mm ?? 0),
        1,
      ),
    );
    const vegTop = niceTop(
      Math.max(...vegetation.map((v) => v.value ?? 0), ...vegetation.map((v) => v.normal ?? 0), 0.2),
    );

    const rainY = (mm: number) => PAD.top + RAIN_H - (mm / rainTop) * RAIN_H;
    const vegBase = PAD.top + RAIN_H + GAP;
    const vegY = (v: number) => vegBase + VEG_H - (v / vegTop) * VEG_H;

    // The normal is a per-calendar-month value, so it is drawn as a step rather
    // than interpolated: the comparison is month against month.
    const stepPath = (
      values: Array<number | null | undefined>,
      y: (v: number) => number,
    ): string => {
      const parts: string[] = [];
      let open: string | null = null;
      months.forEach((m, i) => {
        const value = values[i];
        const calendar = m?.slice(5, 7) ?? null;
        const x0 = PAD.left + i * slot;
        const x1 = x0 + slot;
        if (value == null) return;
        if (open !== calendar) {
          if (parts.length) parts.push(`H ${x1.toFixed(1)}`);
          parts.push(`M ${x0.toFixed(1)} ${y(value).toFixed(1)}`);
          open = calendar;
        }
      });
      if (parts.length) parts.push(`H ${(PAD.left + months.length * slot).toFixed(1)}`);
      return parts.join(" ");
    };

    const vegLine = vegetation
      .map((v, i) => {
        const value = v.value ?? 0;
        if (value == null) return null;
        const x = PAD.left + i * slot + slot / 2;
        return `${i === 0 ? "M" : "L"} ${x.toFixed(1)} ${vegY(value).toFixed(1)}`;
      })
      .filter(Boolean)
      .join(" ");

    const ticks = yearTicks(months);

    return {
      months, plotW, slot, rainTop, vegTop, rainY, vegY, vegBase,
      rainBars: rain.map((r, i) => {
        const value = r.precip_mm ?? 0;
        return {
          i,
          x: PAD.left + i * slot,
          top: rainY(value),
          height: Math.max(value > 0 ? 1 : 0, rainY(0) - rainY(value)),
          dry: r.anomaly_pct != null && r.anomaly_pct <= -50,
          suspect: Boolean(r.suspect),
        };
      }),
      rainStep: stepPath(rain.map((r) => r.normal_mm), rainY),
      vegStep: stepPath(vegetation.map((v) => v.normal), vegY),
      vegLine,
      ticks,
      hasRain: rain.length > 0,
      hasVeg: vegetation.length > 0,
    };
  }, [rain, vegetation]);

  if (!model) return null;

  const active = hover == null ? null : model.months[hover];
  const activeRain = hover != null ? rain[hover] : null;
  const activeVeg = hover != null ? vegetation[hover] : null;

  return (
    <figure className="fig mt-5">
      <svg
        viewBox={`0 0 ${W} ${TOTAL_H}`}
        className="w-full"
        role="img"
        aria-label="Monthly rainfall and vegetation index against their climatological normals"
        onMouseLeave={() => setHover(null)}
      >
        {/* --- rainfall panel --- */}
        {model.hasRain && (
          <>
            <text x={PAD.left} y={PAD.top - 1} className="fill-[var(--ink)] text-[10px] font-medium">
              Rainfall
            </text>
            {[0, 0.5, 1].map((f) => {
              const v = model.rainTop * f;
              return (
                <g key={`r${f}`}>
                  <line
                    x1={PAD.left} x2={W - PAD.right}
                    y1={model.rainY(v)} y2={model.rainY(v)}
                    stroke="var(--rule)" strokeWidth={1}
                  />
                  <text
                    x={PAD.left - 6} y={model.rainY(v) + 3}
                    textAnchor="end" className="fill-[var(--ink-3)] text-[9px]"
                  >
                    {Math.round(v)}
                  </text>
                </g>
              );
            })}
            <text x={PAD.left - 6} y={PAD.top + 8} textAnchor="end" className="fill-[var(--ink-3)] text-[9px]">
              mm
            </text>
            {model.rainBars.map((bar) => (
              <rect
                key={bar.i}
                x={bar.x + (model.slot - Math.min(model.slot * 0.74, 14)) / 2}
                y={bar.top}
                width={Math.max(1, Math.min(model.slot * 0.74, 14))}
                height={bar.height}
                fill={bar.dry ? "var(--caution)" : "var(--ink-2)"}
                opacity={bar.dry ? 1 : 0.72}
                onMouseEnter={() => setHover(bar.i)}
              />
            ))}
            {model.rainStep && (
              <path d={model.rainStep} fill="none" stroke="var(--ink)" strokeWidth={1.25} />
            )}
            <text
              x={W - PAD.right + 6} y={PAD.top + 8}
              className="fill-[var(--ink-2)] text-[9px]"
            >
              {rainNormal ? `normal ${rainNormal.slice(0, 4)}–` : 'normal'}
            </text>
            <text
              x={W - PAD.right + 6} y={PAD.top + 19}
              className="fill-[var(--ink-2)] text-[9px]"
            >
              {rainNormal ? rainNormal.slice(-4) : '—'}
            </text>
          </>
        )}

        {/* --- vegetation panel --- */}
        {model.hasVeg && (
          <>
            <text
              x={PAD.left} y={model.vegBase - 4}
              className="fill-[var(--ink)] text-[10px] font-medium"
            >
              Vegetation index
            </text>
            {[0, 1].map((f) => {
              const v = model.vegTop * f;
              return (
                <g key={`v${f}`}>
                  <line
                    x1={PAD.left} x2={W - PAD.right}
                    y1={model.vegY(v)} y2={model.vegY(v)}
                    stroke="var(--rule)" strokeWidth={1}
                  />
                  <text
                    x={PAD.left - 6} y={model.vegY(v) + 3}
                    textAnchor="end" className="fill-[var(--ink-3)] text-[9px]"
                  >
                    {v.toFixed(1)}
                  </text>
                </g>
              );
            })}
            {model.vegStep && (
              <path d={model.vegStep} fill="none" stroke="var(--ink-3)" strokeWidth={1} strokeDasharray="3 2" />
            )}
            <path d={model.vegLine} fill="none" stroke="var(--accent)" strokeWidth={1.4} />
            <text
              x={W - PAD.right + 6} y={model.vegBase + 4}
              className="fill-[var(--ink-2)] text-[9px]"
            >
              {vegetationNormal ? `normal ${vegetationNormal.slice(0, 4)}–` : 'normal'}
            </text>
            <text
              x={W - PAD.right + 6} y={model.vegBase + 15}
              className="fill-[var(--ink-2)] text-[9px]"
            >
              {vegetationNormal ? vegetationNormal.slice(-4) : '—'}
            </text>
          </>
        )}

        {/* --- shared time axis --- */}
        {model.ticks.map((t, i) => {
          const keep = i % Math.ceil(model.ticks.length / 9) === 0;
          if (!keep) return null;
          return (
            <text
              key={`y${t.year}`}
              x={PAD.left + t.x} y={TOTAL_H - 8}
              textAnchor="start"
              className="fill-[var(--ink-3)] text-[9px]"
            >
              {t.year}
            </text>
          );
        })}

        {/* --- read-out, so no hover is required to know where you are --- */}
        <text x={PAD.left} y={TOTAL_H - 8} className="fill-[var(--ink)] text-[9px]">
          {active ?? (model.months[model.months.length - 1] ?? '')}
        </text>
        {activeRain && (
          <text
            x={PAD.left + 74} y={TOTAL_H - 8}
            className="fill-[var(--ink-2)] text-[9px]"
          >
            {`${Math.round(activeRain.precip_mm)} mm${
              activeRain.anomaly_pct != null
                ? `  ${activeRain.anomaly_pct > 0 ? '+' : ''}${Math.round(activeRain.anomaly_pct)}%`
                : ''
            }`}
          </text>
        )}
        {activeVeg && (
          <text
            x={PAD.left + 168} y={TOTAL_H - 8}
            className="fill-[var(--ink-2)] text-[9px]"
          >
            {`NDVI ${activeVeg.value?.toFixed(3)}`}
          </text>
        )}
      </svg>

      <figcaption className="mt-1 space-y-1 text-[10px] text-[var(--ink-3)]">
        <p>
          Upper: monthly rainfall, mm, bars; the line is that calendar month&rsquo;s
          normal. Lower: area-mean NDVI, the line its normal, on the same months.
          Caution marks a rainfall month at or below 50% of normal.
        </p>
        {vegetationSource && (
          <p>
            Vegetation: {vegetationSource}
            {vegetationStatus ? ` · ${vegetationStatus}` : ''}.
          </p>
        )}
        <p>
          Co-variation with rainfall is not attribution: vegetation responds to rain
          with a lag that varies by season, and water is not always the limiting
          factor in semi-arid rangeland.
        </p>
      </figcaption>
    </figure>
  );
}
