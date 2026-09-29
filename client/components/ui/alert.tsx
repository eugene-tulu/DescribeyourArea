import * as React from 'react';
import { cva, type VariantProps } from 'class-variance-authority';

import { cn } from '@/lib/utils';

/* An alert is a warm lane, not a red one. The old interface shouted in red for
   every refusal, which trained people to ignore the panel entirely; here only a
   genuine failure takes the bare end of the ramp, and even that is a wash rather
   than a block, because most refusals in this product have a route through
   them and the panel has to read as information rather than as a wall. */
const alertVariants = cva(
  'relative w-full rounded-xl border px-4 py-3.5 flex items-start gap-3 [&>svg]:mt-0.5 [&>svg]:shrink-0',
  {
    variants: {
      variant: {
        default: 'border-line-2 bg-raised text-ink-2 [&>svg]:text-ink-3',
        caution:
          'border-stressed/35 bg-stressed/10 text-stressed [&>svg]:text-stressed',
        destructive:
          'border-bare/40 bg-bare/10 text-bare [&>svg]:text-bare',
      },
    },
    defaultVariants: {
      variant: 'default',
    },
  }
);

const Alert = React.forwardRef<
  HTMLDivElement,
  React.HTMLAttributes<HTMLDivElement> & VariantProps<typeof alertVariants>
>(({ className, variant, ...props }, ref) => (
  <div
    ref={ref}
    className={cn(alertVariants({ variant }), className)}
    {...props}
  />
));
Alert.displayName = 'Alert';

const AlertTitle = React.forwardRef<
  HTMLParagraphElement,
  React.HTMLAttributes<HTMLHeadingElement>
>(({ className, ...props }, ref) => (
  <h5
    ref={ref}
    className={cn('mb-1 text-sm font-semibold tracking-tight', className)}
    {...props}
  />
));
AlertTitle.displayName = 'AlertTitle';

const AlertDescription = React.forwardRef<
  HTMLParagraphElement,
  React.HTMLAttributes<HTMLHeadingElement>
>(({ className, ...props }, ref) => (
  <div
    ref={ref}
    className={cn('text-sm leading-relaxed [&_p]:leading-relaxed', className)}
    {...props}
  />
));
AlertDescription.displayName = 'AlertDescription';

export { Alert, AlertTitle, AlertDescription };
