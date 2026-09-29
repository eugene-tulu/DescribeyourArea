import { useState } from "react";
import { Copy } from "lucide-react";
import { Button } from "@/components/ui/button";

export default function CopySummary({ summaryText }: { summaryText: string }) {
  const [copied, setCopied] = useState(false);

  const copyToClipboard = async () => {
    if (summaryText) {
      try {
        await navigator.clipboard.writeText(summaryText);
        setCopied(true);

        // reset after 2s
        setTimeout(() => setCopied(false), 2000);
      } catch (error) {
        console.error("Copy error:", error);
      }
    }
  };

  return (
    <Button onClick={copyToClipboard} variant="outline" size="sm">
      <Copy className="h-3.5 w-3.5" />
      {copied ? "Copied" : "Copy as text"}
    </Button>
  );
}
