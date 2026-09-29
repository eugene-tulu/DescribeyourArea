import * as React from 'react';
import { Slot } from '@radix-ui/react-slot';
import { cva, type VariantProps } from 'class-variance-authority';

import { cn } from '@/lib/utils';

/* The shape language of the app lives in .btn (globals.css) rather than in a
   token soup here, because the interaction -- a hairline that lights, a lift of
   one pixel -- is the part that makes a control feel like a physical instrument
   rather than a web form. This file only decides which weight a control is. */
const buttonVariants = cva('btn', {
  variants: {
    variant: {
      default: 'btn-primary',
      destructive: 'btn-ghost !text-bare !border-bare/40 hover:!bg-bare/10',
      outline: 'btn-ghost',
      secondary: 'btn-ghost',
      ghost: 'btn-quiet',
      link: 'btn-quiet !text-signal hover:underline underline-offset-4',
    },
    size: {
      default: 'h-10 px-4 text-sm',
      sm: 'h-8 px-3 text-[0.8125rem]',
      lg: 'h-12 px-6 text-base',
      icon: 'h-10 w-10',
    },
  },
  defaultVariants: {
    variant: 'default',
    size: 'default',
  },
});

export interface ButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof buttonVariants> {
  asChild?: boolean;
}

const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant, size, asChild = false, ...props }, ref) => {
    const Comp = asChild ? Slot : 'button';
    return (
      <Comp
        className={cn(buttonVariants({ variant, size, className }))}
        ref={ref}
        {...props}
      />
    );
  }
);
Button.displayName = 'Button';

export { Button, buttonVariants };
