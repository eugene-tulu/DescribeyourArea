import * as React from 'react';

import { cn } from '@/lib/utils';

// eslint-disable-next-line @typescript-eslint/no-empty-object-type
export interface InputProps
  extends React.InputHTMLAttributes<HTMLInputElement> {}

const Input = React.forwardRef<HTMLInputElement, InputProps>(
  ({ className, type, ...props }, ref) => {
    return (
      <input
        type={type}
        className={cn(
          // Placeholder sits at text-3, not the old token's muted grey: 4.7:1 is
          // the floor this system holds everywhere, including for text that
          // disappears the moment you start typing.
          'flex h-11 w-full rounded-[10px] border border-rule bg-raised px-3.5 text-sm text-ink',
          'placeholder:text-ink-3',
          'transition-[border-color,box-shadow,background-color] duration-200',
          'hover:border-line-2',
          'focus-visible:border-signal focus-visible:outline-none focus-visible:ring-4 focus-visible:ring-signal/12',
          'disabled:cursor-not-allowed disabled:opacity-50',
          'file:mr-3 file:border-0 file:bg-signal file:px-3 file:py-1.5 file:rounded-md',
          'file:text-[0.8125rem] file:font-semibold file:text-void file:cursor-pointer',
          className
        )}
        ref={ref}
        {...props}
      />
    );
  }
);
Input.displayName = 'Input';

export { Input };
