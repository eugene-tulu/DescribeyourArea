"use client";

import {
  Toast,
  ToastClose,
  ToastDescription,
  ToastProvider,
  ToastTitle,
  ToastViewport,
} from "@/components/ui/toast";
import { useToast } from "@/hooks/use-toast";

/**
 * Mounted once, at the root, and the reason it exists is worth recording.
 *
 * `useToast` dispatched into an in-memory store that nothing rendered, so every
 * message the app produced -- including "invalid GeoJSON file" and "file is too
 * large" -- went nowhere. A rejected upload was not a bad experience, it was an
 * absent one: the user pressed a button and nothing happened, with no way to tell
 * whether the file had loaded.
 *
 * The store is a module singleton, so this component and whoever dispatched are
 * independent. Keeping it at the root means a component three levels down gets
 * its message without a provider threaded down to it.
 */
export function Toaster() {
  const { toasts } = useToast();

  return (
    <ToastProvider swipeDirection="right">
      {toasts.map(({ id, title, description, action, ...props }) => (
        <Toast key={id} {...props}>
          <div className="grid gap-1">
            {title ? <ToastTitle>{title}</ToastTitle> : null}
            {description ? <ToastDescription>{description}</ToastDescription> : null}
          </div>
          {action}
          <ToastClose />
        </Toast>
      ))}
      <ToastViewport />
    </ToastProvider>
  );
}
