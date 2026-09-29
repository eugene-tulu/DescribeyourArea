/* The mark: a 2x2 sampling grid with one cell read.

   It is the raster. Every number this product reports came from a grid of cells
   that were sampled, and this is four of them with one already read. It is also
   the reason the accent is allowed exactly one appearance here: the lit cell is
   the only saturated thing in the top-left corner of the page, which is what
   makes it read as a mark rather than as decoration.

   Drawn as geometry rather than set as a glyph so it is pixel-exact at 14px and
   does not depend on a font being loaded first. */
export function Mark({ size = 18, className }: { size?: number; className?: string }) {
  const cell = size / 3.25;
  const gap = size / 13;
  return (
    <svg
      width={size}
      height={size}
      viewBox={`0 0 ${size} ${size}`}
      fill="none"
      aria-hidden
      className={className}
    >
      <rect x={0} y={0} width={cell} height={cell} rx={cell * 0.22} fill="var(--signal)" />
      <rect
        x={cell + gap}
        y={0}
        width={cell}
        height={cell}
        rx={cell * 0.22}
        fill="var(--signal)"
        opacity={0.22}
      />
      <rect
        x={0}
        y={cell + gap}
        width={cell}
        height={cell}
        rx={cell * 0.22}
        fill="var(--signal)"
        opacity={0.22}
      />
      <rect
        x={cell + gap}
        y={cell + gap}
        width={cell}
        height={cell}
        rx={cell * 0.22}
        fill="var(--signal)"
        opacity={0.12}
      />
    </svg>
  );
}

/* The wordmark. Set in the display serif rather than the UI sans, which is the
   whole reason it reads as a name and not as a label: the interface is set in
   Inter, so anything in Instrument Serif at the same size already looks like
   something chosen rather than something inherited. */
export function Wordmark({ className }: { className?: string }) {
  return (
    <span className={`flex items-center gap-2.5 ${className ?? ''}`}>
      <Mark size={16} />
      <span
        className="font-serif text-[1.0625rem] leading-none tracking-[-0.01em]"
        style={{ fontFamily: '"Instrument Serif", "Iowan Old Style", Georgia, serif' }}
      >
        GeoContextualize
      </span>
    </span>
  );
}
